import dash
from dash import dcc, html, dash_table, callback
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from fredapi import Fred
import dash_bootstrap_components as dbc
import io
import requests

dash.register_page(__name__, path='/yield_curves')

def create_yield_table(data, columns, labels, table_id):
    """Create standardized yield table component for Dash"""
    return html.Div([
        dash_table.DataTable(
            id=table_id,
            columns=[{'name': col, 'id': col} for col in columns],
            data=data.to_dict('records'),
            editable=False,
            filter_action="none",
            row_selectable=False,
            fixed_columns={'headers': True},
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
                'width': 'auto',
                'font-family': 'Arial',
                'color': 'black',
                'font-size': '14px',
            },
            style_cell={
                'padding': '10px',
                'textAlign': 'right',
                'backgroundColor': 'white'
            },
            style_cell_conditional=[{
                'if': {'column_id': 'Date'},
                'textAlign': 'left'
            }] + [{
                'if': {'column_id': col},
                'textAlign': 'left'
            } for col in labels],
            style_header={
                'fontWeight': 'bold',
                'color': 'black',
                'whiteSpace': 'normal',
                'border': '2px solid black',
                'font-family': 'Arial',
                'font-size': '20px'
            },
            style_table={
                'height': 'auto',
                'overflowX': 'auto',
                'width': 'auto'
            }
        )
    ], style={'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px', 'marginBottom': '20px'})

today = datetime.now().date().strftime('%Y-%m-%d')
nor_url = f"https://data.norges-bank.no/api/data/GOVT_GENERIC_RATES/B.7Y+6M+5Y+3Y+3M+12M+10Y.GBON+TBIL.?format=csv&startPeriod=2000-10-17&endPeriod={today}&locale=en"

nor_response = requests.get(nor_url)

if nor_response.status_code == 200:
    nor_csv_data = nor_response.content.decode('utf-8')
    nor_data = pd.read_csv(io.StringIO(nor_csv_data), sep=';')
    nor_data = nor_data[['Tenor', 'TIME_PERIOD', 'OBS_VALUE']].rename(columns={'TIME_PERIOD': 'Date', 'OBS_VALUE': 'Yield'})

nor_yield_df = nor_data.pivot(index='Date', columns='Tenor', values='Yield').reset_index()
nor_yield_df = nor_yield_df[['Date', '3 months', '6 months', '12 months', '3 years', '5 years', '7 years', '10 years']]
nor_yield_df.rename(columns={
    '3 months': '3M', '6 months': '6M', '12 months': '1Y', '3 years': '3Y',
    '5 years': '5Y', '7 years': '7Y', '10 years': '10Y'
}, inplace=True)

nor_yield_df['Date'] = pd.to_datetime(nor_yield_df['Date'])
nor_yield_df.set_index('Date', inplace=True)
nor_yield_df.dropna(how='all', inplace=True)
nor_yield_df.fillna(method='ffill', inplace=True)

nor_yield_monthly = nor_yield_df.resample('ME').last()

nor_today = datetime.today().date()
nor_yield_monthly.sort_index(inplace=True)

nor_yield_monthly_reset = nor_yield_monthly.reset_index()
nor_yield_monthly_reset.sort_values('Date', ascending=False, inplace=True)
nor_yield_monthly_reset['Date'] = nor_yield_monthly_reset['Date'].dt.strftime('%d-%m-%Y')
nor_yield_monthly_reset.iloc[0,0] = nor_today.strftime('%d-%m-%Y')

nor_last_yields = nor_yield_monthly.iloc[-1]
norwegian_labels = ['3M', '6M', '1Y', '3Y', '5Y', '7Y', '10Y']

nor_static_table_content = [
    html.Tr([html.Td('Maturity')] + [html.Td(maturity) for maturity in norwegian_labels]),
    html.Tr([html.Td('Yield (%)')] + [html.Td(f"{nor_last_yields[i]:.2f}") for i in norwegian_labels])
]

nor_static_yield_table = html.Table(
    nor_static_table_content,
    style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid black'},
)

nor_yield_table_header = html.Div(f"Yields as of {nor_today}", style={'fontWeight': 'bold', 'marginBottom': '10px'})

norwegian_yield_curve_layout = html.Div([
    html.H1("Historical Norwegian Yield Curve", style={'textAlign':'center'}),
    html.Div([nor_yield_table_header, nor_static_yield_table], style={'padding': '10px', 'backgroundColor': '#f9f9f9', 
                                                                      'borderRadius': '5px', 'marginBottom': '20px'}),
    dcc.DatePickerRange(
        id='nor-date-picker-range',
        start_date=nor_yield_monthly.index.min().date(),
        end_date=nor_yield_monthly.index.max().date(),
        display_format='YYYY-MM-DD',
        style={'marginBottom': '20px'}
    ),
    dcc.Graph(id='nor-yield-curve-3d', config={'scrollZoom': True}, style={'height': '1000px'}),
    create_yield_table(nor_yield_monthly_reset, nor_yield_monthly_reset.columns, ['Date'], 'nor-yield-table')
], style={'textAlign':'center'})

FRED_API_KEY = '6188d31bebbdca093493a1077d095284'
fred = Fred(FRED_API_KEY)

observation_start = (datetime.today() - timedelta(days=20 * 365)).strftime('%Y-%m-%d')

maturity_labels = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
series_ids = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5',
              'DGS7', 'DGS10', 'DGS20', 'DGS30']

yield_data = {}
for series_id in series_ids:
    yield_data[series_id] = fred.get_series(series_id, observation_start=observation_start)

yield_df = pd.DataFrame(yield_data)
yield_df.index = pd.to_datetime(yield_df.index)

yield_df.dropna(how='all', inplace=True)
yield_df.fillna(method='ffill', inplace=True)

yield_df_quarterly = yield_df.resample('QE').last()

today = pd.Timestamp(datetime.today().date())

yield_df_quarterly.sort_index(inplace=True)
yield_df_quarterly.columns = maturity_labels
table_yields = yield_df_quarterly.reset_index()

table_yields['index'] = pd.to_datetime(table_yields['index']).dt.strftime('%Y-%m-%d')
table_yields.rename(columns={'index':'Date'}, inplace=True)
table_yields.sort_values('Date', ascending=False, inplace=True)
table_yields.iloc[0,0] = nor_today
last_yields = yield_df_quarterly.iloc[-1]
static_table_content = [
    html.Tr([html.Td('Maturity')] + [html.Td(maturity) for maturity in maturity_labels]),
    html.Tr([html.Td('Yield (%)')] + [html.Td(f"{last_yields[i]:.2f}") for i in range(len(maturity_labels))])
]

static_yield_table = html.Table(
    static_table_content,
    style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid black'},
)

yield_table_header = html.Div(f"Yields as of {today.date()}", style={'fontWeight': 'bold', 'marginBottom': '10px', 'textAlign':'center'})

tab1_content = html.Div([
    html.H1("Historical US Yield Curve"),
    html.Div([yield_table_header, static_yield_table], style={'padding': '10px', 'backgroundColor': '#f9f9f9',
                                                               'borderRadius': '5px', 'marginBottom': '20px'}),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=yield_df_quarterly.index.min().date(),
        end_date=datetime.today().date(),
        display_format='YYYY-MM-DD',
        style={'marginBottom': '20px'}
    ),
    dcc.Graph(id='yield-curve-3df', config={'scrollZoom': True}, style={'height': '1000px'}),
    html.Br(),
    create_yield_table(table_yields, table_yields.columns, ['Date'], 'us-yield-table')
], style={'textAlign':'center'})

tab2_content = norwegian_yield_curve_layout

layout_page = dbc.Container([
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='US Yield Curve', value='tab-1', style={'padding': '10px'}),
        dcc.Tab(label='Norwegian Yield Curve', value='tab-2', style={'padding': '10px'}),
    ], style={'marginBottom': '20px'}),
    html.Div(id='tabs-content')
], className='', fluid=True, style={})

layout = dbc.Container([html.Div(className='beforediv'), layout_page],
    className='')

@callback(
    Output('tabs-content', 'children'),
    Input('tabs-example', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        return tab1_content
    elif tab == 'tab-2':
        return norwegian_yield_curve_layout

@callback(
    Output('nor-yield-curve-3d', 'figure'),
    [Input('nor-date-picker-range', 'start_date'),
     Input('nor-date-picker-range', 'end_date')]
)
def update_norwegian_graph(start_date, end_date):
    mask = (nor_yield_monthly.index >= start_date) & (nor_yield_monthly.index <= end_date)
    filtered_data = nor_yield_monthly.loc[mask]

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    num_years = (end_date - start_date).days / 365.25
    
    if num_years <= 1:
        tick_freq = '3M'
    elif 1 < num_years <= 5:
        tick_freq = '6M'
    else:
        tick_freq = '1Y'

    x_ticks = pd.date_range(start=start_date, end=end_date, freq=tick_freq).to_list()

    fig = go.Figure(data=[go.Surface(
        z=filtered_data.T.values,
        x=filtered_data.index,
        y=np.arange(len(norwegian_labels)),
        colorscale='GnBu',
        colorbar_title='Yield (%)',
    )])

    fig.update_layout(
        title=f"Norwegian Yield Curve (3M to 10Y) from {start_date.date()} to {end_date.date()}",
        scene=dict(
            xaxis_title=' ',
            yaxis_title='Maturity',
            zaxis_title='Yield (%)',
            xaxis=dict(
                tickmode='array',
                tickvals=x_ticks,
                ticktext=[d.strftime('%Y-%m-%d') for d in x_ticks],
                backgroundcolor='white'
            ),
            yaxis=dict(
                tickvals=np.arange(len(norwegian_labels)),
                ticktext=norwegian_labels,
                backgroundcolor='white'
            ),
            zaxis=dict(backgroundcolor='white')
        ),
        scene_camera=dict(
            eye=dict(x=1.25, y=1.25, z=1.25),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0)
        ),
        margin=dict(l=0, r=0, b=10, t=40),
        annotations=[
            dict(
                text="Data Source: Norges Bank",
                x=0.0,
                y=0.1,
                align="right",
                xref="paper",
                yref="paper",
                showarrow=False
            )
        ]
    )
    fig.update_layout(showlegend=False)
    fig.update_annotations(font=dict(family="Helvetica", size=12))
    
    return fig

@callback(
    Output('yield-curve-3df', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graph(start_date, end_date):
    mask = (yield_df_quarterly.index >= start_date) & (yield_df_quarterly.index <= end_date)
    filtered_data = yield_df_quarterly.loc[mask]

    fig = go.Figure(data=[go.Surface(
        z=filtered_data.T.values,
        x=filtered_data.index,
        y=np.arange(len(maturity_labels)), 
        colorscale='GnBu',
        colorbar_title='Yield (%)'
    )])

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    num_years = (end_date - start_date).days / 365.25
    
    if num_years <= 1:
        tick_freq = '3M'
    elif 1 < num_years <= 5:
        tick_freq = '6M'
    elif 5 < num_years <= 20:
        tick_freq = '1Y'
    else:
        tick_freq = '5Y'
    
    x_ticks = pd.date_range(start=start_date, end=end_date, freq=tick_freq).to_list()
    
    if today not in x_ticks:
        x_ticks.append(today)

    x_ticks = sorted(x_ticks)  # Sort to maintain order
    x_tick_values = mdates.date2num(x_ticks)  # Convert to numerical format

    # Update layout for the 3D plot
    fig.update_layout(
        title=f"Yield Curve (1-month to 30-year) from {start_date.date()} to {end_date.date()}",
       
        scene=dict(
            xaxis=dict(
                tickmode='array',
                tickvals=x_ticks,
                ticktext=[d.strftime('%Y-%m-%d') for d in x_ticks],
                backgroundcolor='white'
            ), 
            xaxis_title=' ',
            yaxis_title='Maturity',
            zaxis_title='Yield (%)',
            yaxis=dict(
                tickvals=np.arange(len(maturity_labels)), 
                ticktext=maturity_labels,
                backgroundcolor='white'
            ),
            zaxis=dict(backgroundcolor='white')
        ),
        scene_camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)),
    )
    fig.update_layout(showlegend=False)
    
    return fig
