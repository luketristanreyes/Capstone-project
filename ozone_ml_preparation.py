from pathlib import Path
import json
import os
import zipfile
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from IPython.display import display, FileLink


def find_file(name):
    preferred = Path('/kaggle/working') / name
    if preferred.is_file():
        return preferred
    matches = sorted(Path('/kaggle/input').rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f'Attach one copy of {name}; found {len(matches)}.')
    return matches[0]


def load_hourly(path, prefix, variables):
    data = pd.read_csv(path)
    required = ['Site', 'Datetime', 'Parameter', 'Value']
    if not set(required).issubset(data.columns):
        raise ValueError(f'Missing required columns in {path}')
    data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
    data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
    data['Parameter'] = data['Parameter'].astype('string').str.strip().str.upper()
    if data[required].isna().any().any() or not np.isfinite(data.Value).all():
        raise ValueError('Invalid input measurements.')
    if data.Site.nunique() != 1 or data.duplicated(['Datetime','Parameter']).any():
        raise ValueError('Expected one site and unique timestamp/parameter keys.')
    raw = data.pivot(index='Datetime', columns='Parameter', values='Value').sort_index()
    if not set(variables).issubset(raw.columns):
        raise ValueError(f'Missing variables: {set(variables)-set(raw.columns)}')
    scalar = [v for v in variables if v != 'WD']
    # Observation-instant assumption; bins labelled by END time.
    hourly = raw[scalar].resample('h', closed='left', label='right').mean()
    if 'WD' in variables:
        radians = np.deg2rad(raw.WD)
        sinmean = np.sin(radians).resample('h', closed='left', label='right').mean()
        cosmean = np.cos(radians).resample('h', closed='left', label='right').mean()
        strength = np.hypot(sinmean, cosmean)
        angle = np.arctan2(sinmean, cosmean)
        hourly['WD_sin'] = np.sin(angle).where(strength > 1e-6)
        hourly['WD_cos'] = np.cos(angle).where(strength > 1e-6)
    hourly.columns = [f'{prefix}_{v}' for v in hourly.columns]
    return hourly


def prepare_data(queen_path, pat_path, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pat = load_hourly(pat_path, 'PAT', ['O3', 'NOX'])
    qus = load_hourly(queen_path, 'QUS', ['TEMP', 'RH', 'WS', 'WD'])
    predictor_names = ['PAT_NOX', 'QUS_TEMP', 'QUS_RH', 'QUS_WS', 'QUS_WD_sin', 'QUS_WD_cos']
    combined = pd.concat([pat, qus], axis=1).sort_index()
    # Use the common monitoring span; absent hours inside it stay missing.
    needed = predictor_names + ['PAT_O3']
    start = max(combined[v].first_valid_index() for v in needed)
    end = min(combined[v].last_valid_index() for v in needed)
    if start >= end:
        raise ValueError('No common monitoring period.')
    grid = pd.date_range(start, end, freq='h', name='Forecast_issue_time')
    combined = combined.reindex(grid)
    X = pd.DataFrame(index=grid)
    for v in predictor_names:
        X[v + '_at_issue'] = combined[v]
        for lag in [1, 3, 6, 24]:
            X[f'{v}_lag{lag}h'] = combined[v].shift(lag)
    for v in ['PAT_NOX','QUS_TEMP','QUS_RH','QUS_WS']:
        for window in [3, 6, 24]:
            X[f'{v}_recent_{window}h_mean'] = combined[v].rolling(window, min_periods=int(np.ceil(.75*window))).mean()
    # For the one-hour forecast, the target interval starts at issue time.
    hour = grid.hour
    dow = grid.dayofweek
    X['hour_sin'] = np.sin(2*np.pi*hour/24)
    X['hour_cos'] = np.cos(2*np.pi*hour/24)
    X['weekday_sin'] = np.sin(2*np.pi*dow/7)
    X['weekday_cos'] = np.cos(2*np.pi*dow/7)
    X['month_sin'] = np.sin(2*np.pi*(grid.month-1)/12)
    X['month_cos'] = np.cos(2*np.pi*(grid.month-1)/12)
    X['day_of_month'] = grid.day
    X['is_weekend'] = (dow>=5).astype(int)
    season_map = {12:'Summer',1:'Summer',2:'Summer',3:'Autumn',4:'Autumn',5:'Autumn',6:'Winter',7:'Winter',8:'Winter',9:'Spring',10:'Spring',11:'Spring'}
    season = pd.Series(grid.month.map(season_map), index=grid)
    for s in ['Autumn','Winter','Spring']:
        X['season_'+s] = season.eq(s).astype(int)
    # Calendar traffic proxies only, NOT observed traffic/emissions.
    X['weekday_morning_peak_proxy'] = ((dow<5)&(hour>=7)&(hour<=9)).astype(int)
    X['weekday_evening_peak_proxy'] = ((dow<5)&(hour>=16)&(hour<=18)).astype(int)
    X['weekday_business_hours_proxy'] = ((dow<5)&(hour>=9)&(hour<=17)).astype(int)
    # O3 is TARGET ONLY: no O3 predictor columns are generated.
    y = combined.PAT_O3.shift(-1).rename('O3_next_hour_PAT')
    target_time = pd.Series(grid+pd.Timedelta(hours=1), index=grid, name='Target_interval_end')
    # Remove initial lag warm-up; keep gaps thereafter.
    X = X.iloc[24:].replace([np.inf,-np.inf],np.nan)
    y = y.reindex(X.index)
    target_time = target_time.reindex(X.index)
    design = X.copy()
    design['Target_interval_end'] = target_time
    design[y.name] = y
    design.to_csv(out/'full_feature_target_table.csv')
    combined.to_csv(out/'hourly_site_labelled_measurements.csv')
    n = len(X); b1 = int(.60*n); b2 = int(.80*n)
    periods = {'train':X.index[:b1-1], 'validation':X.index[b1:b2-1], 'test':X.index[b2:]}
    sets = {}
    summary = []
    for label, idx in periods.items():
        idx = idx[y.loc[idx].notna().to_numpy()]
        if len(idx)<100:
            raise ValueError(f'Insufficient labelled rows in {label}.')
        sets[label] = (X.loc[idx],y.loc[idx])
        summary.append({'Split':label,'Rows':len(idx),'First_issue_time':idx.min(),'Last_issue_time':idx.max(),
                        'First_target_end':target_time.loc[idx].min(),'Last_target_end':target_time.loc[idx].max()})
    summary = pd.DataFrame(summary)
    summary.to_csv(out/'chronological_split_summary.csv',index=False)
    training = sets['train'][0]
    screen = pd.DataFrame({'Training_missing_fraction':training.isna().mean(),
                           'Training_distinct_values':training.nunique(dropna=True)})
    keep = screen.index[screen.Training_distinct_values>1].tolist()
    screen['Retained'] = screen.index.isin(keep)
    screen.to_csv(out/'training_feature_screening.csv')
    # Fit learned preprocessing on TRAINING ONLY.
    imputer = SimpleImputer(strategy='median', add_indicator=True)
    training_imputed = imputer.fit_transform(training[keep])
    names = imputer.get_feature_names_out(keep)
    scaler = StandardScaler().fit(training_imputed)
    for label,(xx,yy) in sets.items():
        xx = xx[keep]
        imputed = imputer.transform(xx)
        scaled = scaler.transform(imputed)
        assert np.isfinite(scaled).all()
        xx.to_csv(out/f'X_{label}_raw.csv')
        pd.DataFrame(imputed,index=xx.index,columns=names).to_csv(out/f'X_{label}_imputed.csv')
        pd.DataFrame(scaled,index=xx.index,columns=names).to_csv(out/f'X_{label}_scaled.csv')
        yy.to_csv(out/f'y_{label}.csv')
        target_time.loc[xx.index].to_csv(out/f'target_times_{label}.csv')
    joblib.dump({'imputer':imputer,'scaler':scaler,'input_features':keep,'output_features':list(names)},out/'fitted_preprocessing.joblib')
    metadata = {
        'target':'Patumahoe next-hour O3 concentration; source units',
        'predictors':'Patumahoe NOX; Queen Street weather; calendar traffic proxies; temporal features',
        'cross_site_design':True,
        'solar_radiation':'Not supplied; not fabricated',
        'traffic':'Weekday/peak-period calendar proxies, NOT measured counts or emissions',
        'O3_predictor_history_included':False,
        'time_assumptions':'Matching supplied clock conventions; original timestamps are observation instants; hourly bins labelled by end; no measurement-reporting delay. These assumptions require source verification.',
        'forecast':'At t, use completed intervals ending at or before t to predict O3 for (t,t+1h].',
        'split':'Chronological 60/20/20 hourly timeline, one-hour boundary buffers; rows with missing targets excluded after splitting',
        'preprocessing':'Median imputation + missing indicators + optional scaling; fitted on training only',
        'missing_targets':'Never imputed',
        'evaluation':'Sequential one-hour-ahead; later test inputs may use earlier observed test-period measurements, not future values',
        'source_QC':'Unusual WS maximum retained; source verification and sensitivity analysis required',
        'input_features':keep,
        'processed_feature_count':len(names)
    }
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return summary, metadata


if __name__ == '__main__':
    output = Path('/kaggle/working/ozone_ml_ready')
    summary, metadata = prepare_data(find_file('queen_street_combined_clean.csv'),find_file('patumahoe_combined_clean.csv'),output)
    display(summary)
    print('Target:',metadata['target'])
    print('Retained input features:',len(metadata['input_features']))
    print('Processed feature columns:',metadata['processed_feature_count'])
    zip_path = Path('/kaggle/working/ozone_ml_ready.zip')
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.glob('*')):
            if file.is_file():archive.write(file,arcname=file.name)
    os.chdir('/kaggle/working')
    display(FileLink(zip_path.name))
