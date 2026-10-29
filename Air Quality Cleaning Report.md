# Cleaning report

Generated 2026-10-14 08:53

Rows in = Rows out + Removed + Merged (merged rows were averaged into hourly values).

| File | Rows in | Rows out | Removed | Merged |
|---|---:|---:|---:|---:|
| 1C.-Dashboard_data_Patumahoe_filtered.xlsx | 100,271 | 96,131 | 4,140 | 0 |
| 1C.-Dashboard_data_Queen-Street_filtered.xlsx | 180,715 | 139,314 | 39,201 | 2,200 |
| 2C.-Dashboard_data_Patumahoe_filtered.xlsx | 4,392 | 4,181 | 211 | 0 |
| 2C.-Dashboard_data_Queen-Street_filtered_2019-2023.xlsx | 62,710 | 62,621 | 89 | 0 |

## 1C.-Dashboard_data_Patumahoe_filtered.xlsx

| Category | Rule | Removed | Changed | Merged |
|---|---|---:|---:|---:|
| Error | Negative gas concentration | 4,140 | 0 | 0 |

## 1C.-Dashboard_data_Queen-Street_filtered.xlsx

| Category | Rule | Removed | Changed | Merged |
|---|---|---:|---:|---:|
| Error | Unreadable time text (e.g. '1:10:00:00 am am') | 496 | 0 | 0 |
| Duplicate | Exact duplicate row | 37,201 | 0 | 0 |
| Duplicate | Same timestamp, conflicting values (all copies) | 941 | 0 | 0 |
| Inconsistency | Sub-hourly readings averaged to the hour | 0 | 2,231 | 2,200 |
| Error | Placeholder zero (Temp=0 logger outage) | 563 | 0 | 0 |

## 2C.-Dashboard_data_Patumahoe_filtered.xlsx

| Category | Rule | Removed | Changed | Merged |
|---|---|---:|---:|---:|
| Missing | Blank value | 211 | 0 | 0 |

## 2C.-Dashboard_data_Queen-Street_filtered_2019-2023.xlsx

| Category | Rule | Removed | Changed | Merged |
|---|---|---:|---:|---:|
| Inconsistency | Parameter 'TEMP' renamed to 'Temp' | 0 | 728 | 0 |
| Missing | Blank value | 88 | 0 | 0 |
| Error | Outside physically valid range | 1 | 0 | 0 |
