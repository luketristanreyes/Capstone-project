"""Clean Auckland air-quality dashboard extracts.

Removes errors, missing values, duplicates and inconsistencies from the
long-format sheets (Site, Date, Time, Parameter, Value) and writes:

* output/<file>_clean.xlsx   cleaned data + Cleaning_Log sheet
* output/<file>_clean.csv    cleaned data (for Power BI / R / Python)
* output/<file>_removed.csv  every removed row with the reason
* output/all_sites_clean.csv all cleaned files combined
* output/cleaning_report.md  summary of what was changed

Usage:
    python clean_air_quality.py                 # uses config.yaml
    python clean_air_quality.py --config my.yaml
    python clean_air_quality.py --files a.xlsx b.xlsx
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

log = logging.getLogger("aqclean")
KEY = ["Site", "Date", "Time", "Parameter"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@dataclass
class CleaningResult:
    clean: pd.DataFrame
    removed: pd.DataFrame
    steps: list[dict] = field(default_factory=list)


class Cleaner:
    """Runs each cleaning rule and records what it removed or changed."""

    def __init__(self, config: dict):
        self.cfg = config
        self.steps: list[dict] = []
        self.removed: list[pd.DataFrame] = []

    # -- bookkeeping -------------------------------------------------------
    def _remove(self, df: pd.DataFrame, mask: pd.Series, category: str, reason: str) -> pd.DataFrame:
        mask = mask.fillna(False).astype(bool)
        n = int(mask.sum())
        if n:
            gone = df.loc[mask, self.cfg["columns"]].copy()
            gone["Category"] = category
            gone["Reason"] = reason
            self.removed.append(gone)
        self.steps.append({"category": category, "rule": reason, "rows_removed": n,
                           "rows_changed": 0, "rows_merged": 0})
        log.info("  %-15s %-55s removed %7d", category, reason, n)
        return df.loc[~mask].copy()

    def _changed(self, category: str, reason: str, n: int, merged: int = 0) -> None:
        self.steps.append({"category": category, "rule": reason, "rows_removed": 0,
                           "rows_changed": int(n), "rows_merged": int(merged)})
        log.info("  %-15s %-55s changed %7d  merged %7d", category, reason, n, merged)

    # -- pipeline ----------------------------------------------------------
    def run(self, df: pd.DataFrame) -> CleaningResult:
        df = df[self.cfg["columns"]].copy()
        df = self.standardise_text(df)
        df = self.parse_types(df)
        df = self.drop_missing(df)
        df = self.drop_exact_duplicates(df)
        df = self.resolve_conflicting_duplicates(df)
        df = self.resample_hourly(df)
        df = self.handle_negatives(df)
        df = self.drop_out_of_range(df)
        df = self.drop_placeholder_zeros(df)
        df = self.finalise(df)
        removed = (pd.concat(self.removed, ignore_index=True) if self.removed
                   else pd.DataFrame(columns=self.cfg["columns"] + ["Category", "Reason"]))
        return CleaningResult(df, removed, self.steps)

    # 1. Inconsistencies in text fields --------------------------------------
    def standardise_text(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ("Site", "Parameter"):
            df[col] = df[col].astype("string").str.strip()
        before_site = df["Site"].copy()
        df["Site"] = df["Site"].str.upper()
        self._changed("Inconsistency", "Site codes trimmed / upper-cased",
                      (before_site != df["Site"]).sum())

        aliases = {k.upper(): v for k, v in self.cfg["parameter_aliases"].items()}
        before = df["Parameter"].copy()
        df["Parameter"] = df["Parameter"].map(lambda p: aliases.get(str(p).upper(), p) if pd.notna(p) else p)
        changed = (before != df["Parameter"]) & before.notna()
        for old, new in sorted(set(zip(before[changed], df.loc[changed, "Parameter"]))):
            self._changed("Inconsistency", f"Parameter '{old}' renamed to '{new}'",
                          ((before == old) & changed).sum())
        unknown = ~df["Parameter"].isin(set(aliases.values())) & df["Parameter"].notna()
        return self._remove(df, unknown, "Error", "Unknown parameter label")

    # 2. Errors in data types ---------------------------------------------------
    @staticmethod
    def _parse_time(x):
        if isinstance(x, dt.time):
            return x
        if isinstance(x, dt.datetime):
            return x.time()
        if isinstance(x, (int, float)) and not pd.isna(x) and 0 <= x < 1:  # Excel fraction of day
            secs = round(x * 86400)
            return dt.time(secs // 3600, (secs % 3600) // 60)
        if isinstance(x, str):
            s = x.strip().lower()
            for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p"):
                try:
                    return dt.datetime.strptime(s, fmt).time()
                except ValueError:
                    pass
        return None  # unparseable, e.g. "1:10:00:00 am am"

    def parse_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        bad_date = df["Date"].isna()
        df = self._remove(df, bad_date, "Error", "Invalid or missing date")

        parsed = df["Time"].map(self._parse_time)
        bad_time = parsed.isna() & df["Time"].notna()
        df["Time"] = parsed
        df = self._remove(df, bad_time, "Error", "Unreadable time text (e.g. '1:10:00:00 am am')")

        raw = df["Value"]
        df["Value"] = pd.to_numeric(raw, errors="coerce")
        non_numeric = df["Value"].isna() & raw.notna()
        return self._remove(df, non_numeric, "Error", "Non-numeric value")

    # 3. Missing values -----------------------------------------------------------
    def drop_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._remove(df, df["Value"].isna(), "Missing", "Blank value")
        return self._remove(df, df[["Site", "Time", "Parameter"]].isna().any(axis=1),
                            "Missing", "Blank site, time or parameter")

    # 4. Duplicates -------------------------------------------------------------------
    def drop_exact_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._remove(df, df.duplicated(KEY + ["Value"], keep="first"),
                            "Duplicate", "Exact duplicate row")

    def resolve_conflicting_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        dup = df.duplicated(KEY, keep=False)
        policy = self.cfg.get("conflicting_duplicates", "drop")
        if policy == "drop":
            return self._remove(df, dup, "Duplicate",
                                "Same timestamp, conflicting values (all copies)")
        if policy == "first":
            return self._remove(df, df.duplicated(KEY, keep="first"), "Duplicate",
                                "Same timestamp, conflicting values (kept first)")
        if policy == "mean":
            n, before = int(dup.sum()), len(df)
            df = df.groupby(KEY, as_index=False, sort=False)["Value"].mean()
            self._changed("Duplicate", "Same timestamp, conflicting values averaged", n, before - len(df))
            return df
        raise ValueError(f"Unknown conflicting_duplicates policy: {policy}")

    # 5. Inconsistent time resolution ------------------------------------------------
    def resample_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.get("resample_to_hourly", True):
            return df
        sub = df["Time"].map(lambda t: t.minute != 0 or t.second != 0)
        n, before = int(sub.sum()), len(df)
        if n:
            df["Time"] = df["Time"].map(lambda t: dt.time(t.hour))
            df = df.groupby(KEY, as_index=False, sort=False)["Value"].mean()
        self._changed("Inconsistency", "Sub-hourly readings averaged to the hour", n, before - len(df))
        return df

    # 6. Errors: impossible values -----------------------------------------------------
    def handle_negatives(self, df: pd.DataFrame) -> pd.DataFrame:
        policy = self.cfg.get("negative_values", "drop")
        gas = df["Parameter"].isin(self.cfg.get("gas_parameters", []))
        neg = gas & (df["Value"] < 0)
        if policy == "drop":
            return self._remove(df, neg, "Error", "Negative gas concentration")
        if policy == "clip":
            tol = float(self.cfg.get("negative_tolerance", -2.0))
            df = self._remove(df, neg & (df["Value"] < tol), "Error",
                              f"Negative gas concentration below {tol}")
            small = df["Parameter"].isin(self.cfg["gas_parameters"]) & (df["Value"] < 0)
            df.loc[small, "Value"] = 0.0
            self._changed("Error", "Small negative gas concentration set to 0", small.sum())
            return df
        if policy == "keep":
            return df
        raise ValueError(f"Unknown negative_values policy: {policy}")

    def drop_out_of_range(self, df: pd.DataFrame) -> pd.DataFrame:
        keep_neg = self.cfg.get("negative_values") == "keep"
        mask = pd.Series(False, index=df.index)
        for param, (lo, hi) in self.cfg["valid_ranges"].items():
            p = df["Parameter"] == param
            low_bad = df["Value"] < lo
            if keep_neg and param in self.cfg.get("gas_parameters", []):
                low_bad = pd.Series(False, index=df.index)
            mask |= p & (low_bad | (df["Value"] > hi))
        return self._remove(df, mask, "Error", "Outside physically valid range")

    def drop_placeholder_zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        rule = self.cfg.get("placeholder_zero")
        if not rule:
            return df
        main = rule["parameter"]
        zero_main = (df["Parameter"] == main) & (df["Value"] == 0)
        stamps = set(map(tuple, df.loc[zero_main, ["Site", "Date", "Time"]].itertuples(index=False)))
        stamp_of = pd.Series(list(zip(df["Site"], df["Date"], df["Time"])), index=df.index)
        linked = (df["Parameter"].isin(rule.get("linked_parameters", []))
                  & (df["Value"] == 0) & stamp_of.isin(stamps))
        return self._remove(df, zero_main | linked, "Error",
                            f"Placeholder zero ({main}=0 logger outage)")

    # 7. Tidy output -------------------------------------------------------------------
    def finalise(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Datetime"] = df["Date"] + pd.to_timedelta(
            df["Time"].map(lambda t: t.hour * 3600 + t.minute * 60 + t.second), unit="s")
        df = df.sort_values(["Site", "Parameter", "Datetime"]).reset_index(drop=True)
        return df[["Site", "Date", "Time", "Datetime", "Parameter", "Value"]]


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_sheet(path: Path, cfg: dict) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=cfg.get("sheet_name", "Main"))
    missing = set(cfg["columns"]) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    return df


def write_outputs(path: Path, result: CleaningResult, raw_rows: int, out_dir: Path) -> dict:
    stem = re.sub(r"_filtered.*$", "", path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = pd.DataFrame(result.steps)

    result.clean.to_csv(out_dir / f"{stem}_clean.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    result.removed.to_csv(out_dir / f"{stem}_removed.csv", index=False, date_format="%Y-%m-%d")

    xlsx = out_dir / f"{stem}_clean.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl", date_format="dd/mm/yyyy",
                        datetime_format="dd/mm/yyyy hh:mm") as xw:
        result.clean.to_excel(xw, sheet_name="Main", index=False)
        steps.to_excel(xw, sheet_name="Cleaning_Log", index=False)
        ws = xw.sheets["Main"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for col, width in zip("ABCDEF", (8, 12, 10, 18, 11, 10)):
            ws.column_dimensions[col].width = width
        for cell in ws["C"][1:]:
            cell.number_format = "hh:mm"
        log_ws = xw.sheets["Cleaning_Log"]
        for col, width in zip("ABCDE", (15, 60, 14, 14, 14)):
            log_ws.column_dimensions[col].width = width

    return {"file": path.name, "output": xlsx.name, "rows_in": raw_rows,
            "rows_out": len(result.clean), "rows_removed": len(result.removed),
            "rows_merged": int(steps["rows_merged"].sum()),
            "steps": steps}


def write_report(summaries: list[dict], out_dir: Path) -> None:
    lines = ["# Cleaning report", "",
             f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}", "",
             "Rows in = Rows out + Removed + Merged (merged rows were averaged into hourly values).", "",
             "| File | Rows in | Rows out | Removed | Merged |", "|---|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['file']} | {s['rows_in']:,} | {s['rows_out']:,} | "
                     f"{s['rows_removed']:,} | {s['rows_merged']:,} |")
    for s in summaries:
        lines += ["", f"## {s['file']}", "", "| Category | Rule | Removed | Changed | Merged |",
                  "|---|---|---:|---:|---:|"]
        for r in s["steps"].itertuples():
            if r.rows_removed or r.rows_changed or r.rows_merged:
                lines.append(f"| {r.category} | {r.rule} | {r.rows_removed:,} | "
                             f"{r.rows_changed:,} | {r.rows_merged:,} |")
    (out_dir / "cleaning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml", type=Path)
    ap.add_argument("--files", nargs="*", type=Path, help="Excel files (default: all in input_dir)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    base = args.config.resolve().parent
    files = args.files or sorted((base / cfg["input_dir"]).glob("*.xlsx"))
    out_dir = base / cfg["output_dir"]
    if not files:
        log.error("No .xlsx files found in %s", base / cfg["input_dir"])
        return 1

    summaries, combined = [], []
    for path in files:
        log.info("\n%s", path.name)
        raw = read_sheet(path, cfg)
        result = Cleaner(cfg).run(raw)
        summaries.append(write_outputs(path, result, len(raw), out_dir))
        combined.append(result.clean.assign(Source=path.name))
        log.info("  -> %d rows in, %d rows out", len(raw), len(result.clean))

    all_df = pd.concat(combined, ignore_index=True)
    all_df.to_csv(out_dir / "all_sites_clean.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    write_report(summaries, out_dir)
    log.info("\nDone. Outputs written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
