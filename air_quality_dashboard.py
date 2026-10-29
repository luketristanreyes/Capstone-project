from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import ipywidgets as W
from IPython.display import display, HTML, clear_output
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pio.renderers.default = 'notebook'  # Embedded Plotly JS; no CDN needed
NAMES = ['queen_street_combined_clean.csv', 'patumahoe_combined_clean.csv']
LABELS = {'QUS': 'Queen Street', 'PAT': 'Patumahoe'}
POL = ['O3', 'NOX', 'NO2']
MET = ['TEMP', 'RH', 'WS', 'WD']
frames = []
for name in NAMES:
    matches = list(Path('/kaggle/input').rglob(name)) if Path('/kaggle/input').exists() else []
    local = [Path('/kaggle/working') / name, Path(name)]
    path = next((p for p in local if p.exists()), None)
    if path is None:
        if len(matches) != 1:
            raise FileNotFoundError(f'Attach exactly one copy of {name}; found {len(matches)} in Kaggle inputs.')
        path = matches[0]
    frames.append(pd.read_csv(path))
raw = pd.concat(frames, ignore_index=True)
raw['Datetime'] = pd.to_datetime(raw['Datetime'], errors='coerce')
raw['Value'] = pd.to_numeric(raw['Value'], errors='coerce')
raw['Parameter'] = raw['Parameter'].str.strip().str.upper()
invalid_rows = raw['Datetime'].isna().sum()
duplicate_rows = raw.duplicated(['Site', 'Datetime', 'Parameter']).sum()
raw = raw.dropna(subset=['Datetime', 'Site', 'Parameter'])

# Daily means, hourly-availability proxies, and previous-data anomaly flags.
parts, quality_parts = [], []
for (site, parameter), g in raw.groupby(['Site', 'Parameter']):
    s = g.groupby('Datetime')['Value'].mean().sort_index()
    hourly = s.resample('h').mean()
    means = hourly.resample('D').mean()
    observed = hourly.notna().resample('D').sum()
    expected = hourly.resample('D').size()
    q = pd.DataFrame({'Datetime': means.index, 'Site': site, 'Parameter': parameter,
                      'ObservedHours': observed.values, 'ExpectedHours': expected.values})
    q['CoveragePct'] = 100*q['ObservedHours']/q['ExpectedHours']
    quality_parts.append(q)
    if parameter == 'WD':
        # Wind directions need circular rather than arithmetic averaging.
        rad = np.deg2rad(s.dropna())
        means = (np.degrees(np.arctan2(np.sin(rad).resample('D').mean(),
                                      np.cos(rad).resample('D').mean())) + 360) % 360
    part = means.rename('Value').reset_index()
    part['Site'], part['Parameter'] = site, parameter
    parts.append(part)
quality = pd.concat(quality_parts, ignore_index=True)
long = pd.concat(parts, ignore_index=True)
wide = long.pivot(index=['Site', 'Datetime'], columns='Parameter', values='Value').reset_index()
for c in POL + MET:
    if c not in wide:
        wide[c] = np.nan
long['Anomaly'] = False
for _, inds in long.groupby(['Site', 'Parameter']).groups.items():
    g = long.loc[inds].sort_values('Datetime')
    previous = g['Value'].shift(1)
    mu = previous.rolling(30, min_periods=10).mean()
    sd = previous.rolling(30, min_periods=10).std().replace(0, np.nan)
    long.loc[g.index, 'Anomaly'] = ((g['Value']-mu).abs()/sd > 3).fillna(False)

# Daily, one-day-ahead models: past pollutant values + calendar only.
FEATURES = ['lag1', 'lag7', 'mean7', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos']
def make_features(s):
    x = pd.DataFrame(index=s.index)
    x['lag1'] = s.shift(1)
    x['lag7'] = s.shift(7)
    x['mean7'] = s.shift(1).rolling(7, min_periods=1).mean()
    x['month_sin'] = np.sin(2*np.pi*x.index.month/12)
    x['month_cos'] = np.cos(2*np.pi*x.index.month/12)
    x['dow_sin'] = np.sin(2*np.pi*x.index.dayofweek/7)
    x['dow_cos'] = np.cos(2*np.pi*x.index.dayofweek/7)
    return x
models, predictions, metric_rows, forecasts = {}, [], [], []
for (site, parameter), g in long[long['Parameter'].isin(POL)].groupby(['Site', 'Parameter']):
    s = g.set_index('Datetime')['Value'].asfreq('D')
    X = make_features(s)
    frame = X.assign(actual=s).dropna()
    if len(frame) < 100:
        continue
    # Split by calendar date, not randomly. No future observations in features.
    split_date = s.index[int(len(s)*0.8)]
    train, test = frame[frame.index < split_date], frame[frame.index >= split_date]
    if len(train) < 60 or len(test) < 10:
        continue
    model = RandomForestRegressor(n_estimators=80, min_samples_leaf=5,
                                  max_depth=12, random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train['actual'])
    pred = model.predict(test[FEATURES])
    baseline = test['lag1'].values
    p = pd.DataFrame({'Datetime': test.index, 'Site': site, 'Parameter': parameter,
                      'Actual': test['actual'].values, 'Predicted': pred,
                      'Persistence': baseline})
    predictions.append(p)
    for label, yhat in [('Random forest', pred), ('Persistence (previous day)', baseline)]:
        metric_rows.append({'Site': site, 'Parameter': parameter, 'Model': label,
                            'N': len(test), 'MAE': mean_absolute_error(test['actual'], yhat),
                            'RMSE': np.sqrt(mean_squared_error(test['actual'], yhat)),
                            'R2': r2_score(test['actual'], yhat),
                            'TestStart': test.index.min(), 'TestEnd': test.index.max()})
    if parameter == 'O3':
        # Refit on all eligible historical rows only AFTER holdout evaluation.
        model.fit(frame[FEATURES], frame['actual'])
        hist = s.loc[:s.last_valid_index()].copy()
        for step in range(7):
            dt = hist.index[-1] + pd.Timedelta(days=1)
            candidate = hist.reindex(hist.index.append(pd.DatetimeIndex([dt])))
            x = make_features(candidate).loc[[dt], FEATURES]
            if x.isna().any(axis=None):
                warnings.warn(f'{site}: insufficient recent O3 history for a recursive forecast.')
                break
            value = float(model.predict(x)[0])
            forecasts.append({'Datetime': dt, 'Site': site, 'ForecastO3': value})
            hist.loc[dt] = value
predictions = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
metrics = pd.DataFrame(metric_rows)
forecast = pd.DataFrame(forecasts)

# Export reusable analytical outputs.
OUT = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('dashboard_outputs')
OUT.mkdir(exist_ok=True)
for name, data in [('daily_air_quality.csv',long), ('data_quality_proxy.csv',quality),
                   ('holdout_predictions.csv',predictions), ('model_metrics.csv',metrics),
                   ('o3_forecast.csv',forecast)]:
    data.to_csv(OUT/name, index=False)

start = W.DatePicker(description='From:', value=long['Datetime'].min().date())
end = W.DatePicker(description='To:', value=long['Datetime'].max().date())
site = W.SelectMultiple(options=[(LABELS.get(s,s),s) for s in sorted(wide['Site'].unique())],
                       value=tuple(sorted(wide['Site'].unique())), description='Location:')
pollutant = W.SelectMultiple(options=[('O3','O3'),('NOx','NOX'),('NO2','NO2')],
                            value=('O3','NOX','NO2'), description='Pollutants:')
env = W.Dropdown(options=[('No environmental filter','None'),('Temperature','TEMP'),
                         ('Humidity','RH'),('Wind speed','WS'),('Wind direction','WD')],
                 description='Condition:', style={'description_width':'initial'})
low = W.FloatText(value=0, description='Minimum:')
high = W.FloatText(value=30, description='Maximum:')
apply = W.Button(description='Update dashboard', button_style='primary', icon='refresh')
outputs = [W.Output() for _ in range(6)]
tabs = W.Tab(children=outputs)
for i,title in enumerate(['Trends','Site / time patterns','Weather','Predictions / metrics','O3 forecast','Anomalies / quality']):
    tabs.set_title(i,title)
status = W.HTML()
def note(text):
    display(HTML('<p>'+text+'</p>'))
def show(fig):
    fig.update_layout(template='plotly_white', height=400, margin=dict(t=65,b=45),
                      legend_title_text='', hovermode='closest')
    fig.show()
def refresh(_=None):
    for out in outputs:
        out.clear_output(wait=True)
    if start.value is None or end.value is None or start.value > end.value or not site.value or not pollutant.value:
        status.value = 'Select a valid date range, location and pollutant.'
        return
    begin, finish = pd.Timestamp(start.value), pd.Timestamp(end.value)+pd.Timedelta(days=1)
    base = wide[(wide['Datetime']>=begin)&(wide['Datetime']<finish)&wide['Site'].isin(site.value)].copy()
    if env.value != 'None':
        if low.value > high.value:
            status.value = 'Environmental minimum must not exceed maximum.'
            return
        base = base[base[env.value].between(low.value, high.value)]
    keys = base[['Site','Datetime']]
    selected = long.merge(keys,on=['Site','Datetime'],how='inner')
    selected = selected[selected['Parameter'].isin(pollutant.value)]
    valid = selected.dropna(subset=['Value'])
    suffix = ' Missing weather values are excluded when a weather filter is active; Patumahoe has no local weather in these files.' if env.value!='None' else ''
    status.value = f'{len(valid):,} daily pollutant values. Daily means; units are not supplied in the CSVs.{suffix}'
    with outputs[0]:
        if valid.empty:
            note('No matching pollutant data. Change the filters.')
        else:
            display(valid.groupby(['Site','Parameter'])['Value'].agg(['count','mean','min','max']).round(2))
            show(px.line(selected,x='Datetime',y='Value',color='Site',facet_row='Parameter',
                         title='Daily pollutant trends (gaps are not filled)').update_yaxes(matches=None))
    with outputs[1]:
        note('Spatial view is a site comparison, not a map: the files contain no coordinates. NO2 is the only pollutant shared by both sites.')
        if not valid.empty:
            means = valid.groupby(['Site','Parameter'],as_index=False)['Value'].mean()
            show(px.bar(means,x='Site',y='Value',color='Site',facet_col='Parameter',title='Site averages over selected dates').update_yaxes(matches=None))
            monthly = valid.assign(Month=valid['Datetime'].dt.month).groupby(['Site','Parameter','Month'],as_index=False)['Value'].mean()
            show(px.line(monthly,x='Month',y='Value',color='Site',facet_row='Parameter',markers=True,title='Seasonal pattern: calendar-month averages').update_yaxes(matches=None))
            p = valid[valid['Parameter']==pollutant.value[0]].copy()
            if not p.empty:
                p['MonthStart']=p['Datetime'].dt.to_period('M').dt.to_timestamp()
                hm=p.groupby(['Site','MonthStart'])['Value'].mean().unstack('MonthStart')
                show(px.imshow(hm,aspect='auto',labels={'color':'Mean concentration'},title=f'Monthly heatmap: {pollutant.value[0]}'))
    with outputs[2]:
        met = base.melt(id_vars=['Site','Datetime'],value_vars=MET,var_name='Condition',value_name='Value').dropna(subset=['Value'])
        note('Weather is available at Queen Street only. Wind direction uses a circular daily mean. Units must be checked against source documentation.')
        if met.empty:
            note('No meteorological measurements match these filters.')
        else:
            show(px.line(met,x='Datetime',y='Value',color='Site',facet_row='Condition',title='Meteorological conditions').update_yaxes(matches=None))
    with outputs[3]:
        note('Fixed chronological holdout: last 20% of calendar dates per series. One-day-ahead predictions use actual past readings. Changing filters does not retrain the model. Models use pollutant lags and calendar features, not weather.')
        if predictions.empty:
            note('Insufficient data for model evaluation.')
        else:
            p=predictions.merge(keys,on=['Site','Datetime'],how='inner')
            p=p[p['Parameter'].isin(pollutant.value)]
            if p.empty:
                note('No held-out predictions in the selected range. Extend the end date.')
            else:
                rows=[]
                for (s,pol),g in p.groupby(['Site','Parameter']):
                    for col in ['Predicted','Persistence']:
                        rows.append({'Site':s,'Pollutant':pol,'Model':col,'N':len(g),
                                     'MAE':mean_absolute_error(g.Actual,g[col]),
                                     'RMSE':np.sqrt(mean_squared_error(g.Actual,g[col])),
                                     'R2':r2_score(g.Actual,g[col]) if len(g)>1 else np.nan})
                display(pd.DataFrame(rows).round(3))
                p=p.melt(id_vars=['Site','Datetime','Parameter'],value_vars=['Actual','Predicted','Persistence'],var_name='Series',value_name='Value')
                p['Line']=p['Site']+' / '+p['Series']
                show(px.line(p,x='Datetime',y='Value',color='Line',facet_row='Parameter',title='Held-out actual vs predicted').update_yaxes(matches=None))
    with outputs[4]:
        note('Seven-day recursive O3 baseline forecast after the last available O3 observation, not after today or the selected end date. Forecasts are historical, not current operational forecasts. This panel uses the location and O3 selection only; date/weather filters do not change its forecast origin. No validated uncertainty interval is supplied.')
        if forecast.empty or 'O3' not in pollutant.value:
            note('Select O3 and Patumahoe to view the O3 forecast.')
        else:
            f=forecast[forecast['Site'].isin(site.value)]
            if f.empty:
                note('No O3 forecast for the selected location.')
            else:
                display(f.round(3))
                history=long[(long['Parameter']=='O3') & long['Site'].isin(f['Site'])].dropna(subset=['Value'])
                history=history[history['Datetime']>=f['Datetime'].min()-pd.Timedelta(days=30)]
                fig=go.Figure()
                for s,g in history.groupby('Site'):
                    fig.add_trace(go.Scatter(x=g.Datetime,y=g.Value,name=s+' observed',mode='lines'))
                for s,g in f.groupby('Site'):
                    fig.add_trace(go.Scatter(x=g.Datetime,y=g.ForecastO3,name=s+' forecast',mode='lines+markers',line=dict(dash='dash')))
                fig.update_layout(title='O3: recent history and 7-day forecast',yaxis_title='Concentration (source units)')
                show(fig)
    with outputs[5]:
        note('Screening indicators, not sensor certification. Anomalies exceed 3 standard deviations from the previous 30 daily means (at least 10). High values may be real pollution events. Coverage = hours with a reading / hours in the available parameter span; it is a proxy, not verified uptime. Original invalid/duplicate rows: '+str(invalid_rows)+' / '+str(duplicate_rows)+'.')
        if not valid.empty:
            display(valid[valid['Anomaly']][['Site','Datetime','Parameter','Value']].sort_values('Datetime',ascending=False).head(100))
            show(px.scatter(valid,x='Datetime',y='Value',color='Anomaly',symbol='Site',facet_row='Parameter',title='Statistical anomaly screening').update_yaxes(matches=None))
        q=quality.merge(keys,on=['Site','Datetime'],how='inner')
        q=q[q['Parameter'].isin(pollutant.value)]
        if not q.empty:
            summary=q.groupby(['Site','Parameter'])[['ObservedHours','ExpectedHours']].sum()
            summary['CoveragePct']=100*summary['ObservedHours']/summary['ExpectedHours']
            display(summary.round(2))
            show(px.line(q,x='Datetime',y='CoveragePct',color='Site',facet_row='Parameter',title='Daily hourly-availability proxy (%)'))
        note('Channels never supplied for a site are unavailable, not counted as zero coverage. Missing periods before the first or after the last recorded value cannot be diagnosed from these cleaned files.')
apply.on_click(refresh)
display(HTML('<h2>Auckland Air Quality Dashboard</h2><p>Historical daily analysis | O3, NOx, NO2 | Queen Street and Patumahoe</p>'))
display(W.VBox([W.HBox([start,end]), W.HBox([site,pollutant]), W.HBox([env,low,high]), apply,status,tabs]))
refresh()
