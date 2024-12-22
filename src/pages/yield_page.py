import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from fredapi import Fred
import dash_bootstrap_components as dbc
dash.register_page(__name__, path='/yield_curves')

import dash_table
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import dash_bootstrap_components as dbc
import io
import requests

today = datetime.now().date().strftime('%Y-%m-%d')
# The API URL for Norwegian Yield Curve
nor_url = f"https://data.norges-bank.no/api/data/GOVT_GENERIC_RATES/B.7Y+6M+5Y+3Y+3M+12M+10Y.GBON+TBIL.?format=csv&startPeriod=2000-10-17&endPeriod={today}&locale=en"

# Send a GET request to the API
nor_response = requests.get(nor_url)

# Check if the request was successful
if nor_response.status_code == 200:
    # Load the CSV data into a pandas DataFrame
    nor_csv_data = nor_response.content.decode('utf-8')
    nor_data = pd.read_csv(io.StringIO(nor_csv_data), sep=';')

    # Select and rename relevant columns
    nor_data = nor_data[['Tenor', 'TIME_PERIOD', 'OBS_VALUE']].rename(columns={'TIME_PERIOD': 'Date', 'OBS_VALUE': 'Yield'})

# Pivot and reformat Norwegian yields data
nor_yield_df = nor_data.pivot(index='Date', columns='Tenor', values='Yield').reset_index()
nor_yield_df = nor_yield_df[['Date', '3 months', '6 months', '12 months', '3 years', '5 years', '7 years', '10 years']]
nor_yield_df.rename(columns={
    '3 months': '3M', '6 months': '6M', '12 months': '1Y', '3 years': '3Y',
    '5 years': '5Y', '7 years': '7Y', '10 years': '10Y'
}, inplace=True)

# Clean up date formatting and set index
nor_yield_df['Date'] = pd.to_datetime(nor_yield_df['Date'])
nor_yield_df.set_index('Date', inplace=True)
nor_yield_df.dropna(how='all', inplace=True)
nor_yield_df.fillna(method='ffill', inplace=True)

# Resample to monthly frequency
nor_yield_monthly = nor_yield_df.resample('ME').last()


# Update with today's data
nor_today = datetime.today().date()
# if not nor_yield_monthly.index.empty and nor_yield_monthly.index[-1] > nor_today:
#     nor_last_yields = nor_yield_monthly.iloc[-1]
#     nor_yield_monthly.loc[nor_today] = nor_last_yields
nor_yield_monthly.sort_index(inplace=True)


# Create Dash DataTable for historical yields
nor_yield_monthly_reset = nor_yield_monthly.reset_index()
nor_yield_monthly_reset.sort_values('Date', ascending=False, inplace=True)
nor_yield_monthly_reset['Date'] = nor_yield_monthly_reset['Date'].dt.strftime('%d-%m-%Y')

# For display in a table
nor_last_yields = nor_yield_monthly.iloc[-1]
norwegian_labels = ['3M', '6M', '1Y', '3Y', '5Y', '7Y', '10Y']

# Static yield table
nor_static_table_content = [
    html.Tr([html.Td('Maturity')] + [html.Td(maturity) for maturity in norwegian_labels]),  # Header row
    html.Tr([html.Td('Yield (%)')] + [html.Td(f"{nor_last_yields[i]:.2f}") for i in norwegian_labels])  # Yield values
]

# Static yield table component
nor_static_yield_table = html.Table(
    nor_static_table_content,
    style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid black'},
)

# Yield table header
nor_yield_table_header = html.Div(f"Yields as of {nor_today}", style={'fontWeight': 'bold', 'marginBottom': '10px'})

# Define the layout for the Norwegian Yield Curve page (2nd Tab)
norwegian_yield_curve_layout = html.Div([
    html.H1("Historical Norwegian Yield Curve", style={'textAlign':'center'}),

    # Static yield table
    html.Div([nor_yield_table_header, nor_static_yield_table], style={'padding': '10px', 'backgroundColor': '#f9f9f9', 
                                                                      'borderRadius': '5px', 'marginBottom': '20px'}),
    
    # Date range picker
    dcc.DatePickerRange(
        id='nor-date-picker-range',
        start_date=nor_yield_monthly.index.min().date(),
        end_date=nor_yield_monthly.index.max().date(),
        display_format='YYYY-MM-DD',
        style={'marginBottom': '20px'}
    ),
    dcc.Graph(id='nor-yield-curve-3d', config={'scrollZoom': True}, style={'height': '1000px'}),
    
    html.Div([
        dash_table.DataTable(
            id='yield-table',
            columns=[{'name': col, 'id': col} for col in nor_yield_monthly_reset.columns],
            data=nor_yield_monthly_reset.to_dict('records'),
            editable=False,
            filter_action="none",
            # sort_action="native",
            # sort_mode="multi",
            column_selectable="single",
            row_selectable=False,
            row_deletable=False,
            # page_current=0,
            # page_size=10,
            # selected_columns=[],
            # selected_rows=[],
            #scrollable = True,
            # striped=True,
            # virtualization=True,
            # page_action="native",
            fixed_columns={'headers' : True},
            # page_current= 2,
            # page_size= 5,
            # style_as_list_view=True,
            # fill_width=False,
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
                'width': 'auto',
                'font-family': ['Arial'],
                'color': 'black',
                'font-size':'14px',
            },
            style_cell={
                'padding': '10px',
                'textAlign': 'right',
                'backgroundColor': 'white'
            },
            style_cell_conditional=[{
                'if': {
                    'column_id': 'Date'
                },
                'textAlign': 'left',
                #'backgroundColor': '#9fa4d8'
            }] + [{
                'if': {
                    'column_id': c
                },
                'textAlign': 'left',
                'backgroundColor': 'white'
            } for c in ['Ticker', 'Company', 'Rank']],

            style_header={
                #'backgroundColor': colors['accent'],
                'fontWeight': 'bold',
                'color': 'black',
                'whiteSpace': 'normal',
                'border': '2px solid black',
                'font-family': ['Arial'],
                'font-size':'20px'
            },
            style_data_conditional=[  # style_data.c refers only to data rows
                # {
                #    'if': {
                #        'row_index': 'odd'
                #    },
                #    'backgroundColor': 'white'
                # },
                {
                    'if': {
                        'column_id': 'Date'
                    },
                    # 'backgroundColor': 'grey',
                    'fontWeight': 'bold',
                },
                {
                    'if': {
                        'column_id': 'Combined Score'
                    },
                    # 'backgroundColor': 'grey',
                    'fontWeight': 'bold',
                }

            ],
            style_table={
                'height': 'auto',
                'overflowX': 'auto',
                # 'overflowY': 'None',
                'width': 'auto',
                # 'font-family': ['Open Sans', 'sans-serif']
            },
            style_filter={'textAlign': 'center', 'font-style': ['bold'],
                            'font-family': ['Arial']}  
              # Align Date column to the left
        )
    ], style={'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px', 'marginBottom': '20px'}),



], style={'textAlign':'center'})

# Initialize FRED API
FRED_API_KEY = '6188d31bebbdca093493a1077d095284'
fred = Fred(FRED_API_KEY)

# Define the observation period (e.g., last 10 years)
observation_start = (datetime.today() - timedelta(days=20 * 365)).strftime('%Y-%m-%d')  # 10 years ago

# List of maturities and their corresponding FRED series IDs
maturity_labels = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
series_ids = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5',
              'DGS7', 'DGS10', 'DGS20', 'DGS30']

# Fetch data for each series over the entire historical period
yield_data = {}
for series_id in series_ids:
    yield_data[series_id] = fred.get_series(series_id, observation_start=observation_start)

# Combine all series into a single DataFrame
yield_df = pd.DataFrame(yield_data)
yield_df.index = pd.to_datetime(yield_df.index)

# Drop rows with all NaN values and fill forward to ensure continuous data
yield_df.dropna(how='all', inplace=True)
yield_df.fillna(method='ffill', inplace=True)

# Resample data to quarterly frequency and include the last observation
yield_df_quarterly = yield_df.resample('QE').last()

# Update the last date's yield values to today's date
today = pd.Timestamp(datetime.today().date())

# If today's date already exists in the DataFrame, keep the values intact
yield_df_quarterly.sort_index(inplace=True)
yield_df_quarterly.columns = maturity_labels
table_yields = yield_df_quarterly.reset_index()

table_yields['index'] = pd.to_datetime(table_yields['index']).dt.strftime('%Y-%m-%d')
table_yields.rename(columns={'index':'Date'}, inplace=True)
table_yields.sort_values('Date', ascending=False, inplace=True)
# Initialize static yield table data
last_yields = yield_df_quarterly.iloc[-1]  # Get the latest yields
static_table_content = [
    html.Tr([html.Td('Maturity')] + [html.Td(maturity) for maturity in maturity_labels]),  # Header row
    html.Tr([html.Td('Yield (%)')] + [html.Td(f"{last_yields[i]:.2f}") for i in range(len(maturity_labels))])  # Yield values
]

# Create the static yield table
static_yield_table = html.Table(
    static_table_content,
    style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid black'},
)

# Add header for the yields
yield_table_header = html.Div(f"Yields as of {today.date()}", style={'fontWeight': 'bold', 'marginBottom': '10px', 'textAlign':'center'})

# Layout for the first tab (the one you already have)
tab1_content = html.Div([
    html.H1("Historical US Yield Curve"),
    
    # Static yield table
    html.Div([yield_table_header, static_yield_table], style={'padding': '10px', 'backgroundColor': '#f9f9f9',
                                                               'borderRadius': '5px', 'marginBottom': '20px'}),
    
    # Date range picker
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=yield_df_quarterly.index.min().date(),
        end_date=datetime.today().date(),
        display_format='YYYY-MM-DD',
        style={'marginBottom': '20px'}
    ),
    
    # 3D Yield Curve Graph
    dcc.Graph(id='yield-curve-3df', config={'scrollZoom': True}, style={'height': '1000px'}),
    html.Br(),
     html.Div([
        dash_table.DataTable(
            id='yield-table',
            columns=[{'name': col, 'id': col} for col in table_yields.columns],
            data=table_yields.to_dict('records'),
            editable=False,
            filter_action="none",
            # sort_action="native",
            # sort_mode="multi",
            column_selectable="single",
            row_selectable=False,
            row_deletable=False,
            # page_current=0,
            # page_size=10,
            # selected_columns=[],
            # selected_rows=[],
            #scrollable = True,
            # striped=True,
            # virtualization=True,
            # page_action="native",
            fixed_columns={'headers' : True},
            # page_current= 2,
            # page_size= 5,
            # style_as_list_view=True,
            # fill_width=False,
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
                'width': 'auto',
                'font-family': ['Arial'],
                'color': 'black',
                'font-size':'14px',
            },
            style_cell={
                'padding': '10px',
                'textAlign': 'right',
                'backgroundColor': 'white'
            },
            style_cell_conditional=[{
                'if': {
                    'column_id': 'Date'
                },
                'textAlign': 'left',
                #'backgroundColor': '#9fa4d8'
            }] + [{
                'if': {
                    'column_id': c
                },
                'textAlign': 'left',
                'backgroundColor': 'white'
            } for c in ['Ticker', 'Company', 'Rank']],

            style_header={
                #'backgroundColor': colors['accent'],
                'fontWeight': 'bold',
                'color': 'black',
                'whiteSpace': 'normal',
                'border': '2px solid black',
                'font-family': ['Arial'],
                'font-size':'20px'
            },
            style_data_conditional=[  # style_data.c refers only to data rows
                # {
                #    'if': {
                #        'row_index': 'odd'
                #    },
                #    'backgroundColor': 'white'
                # },
                {
                    'if': {
                        'column_id': 'Date'
                    },
                    # 'backgroundColor': 'grey',
                    'fontWeight': 'bold',
                },
                {
                    'if': {
                        'column_id': 'Combined Score'
                    },
                    # 'backgroundColor': 'grey',
                    'fontWeight': 'bold',
                }

            ],
            style_table={
                'height': 'auto',
                'overflowX': 'auto',
                # 'overflowY': 'None',
                'width': 'auto',
                # 'font-family': ['Open Sans', 'sans-serif']
            },
            style_filter={'textAlign': 'center', 'font-style': ['bold'],
                            'font-family': ['Arial']}  
              # Align Date column to the left
        )
    ], style={'padding': '10px', 'backgroundColor': '#f9f9f9', 'marginBottom': '20px'})
], style={'textAlign':'center'})

# Assuming you want to add another page or data visualization to the second tab:
tab2_content = norwegian_yield_curve_layout

# Define the layout of the app with Tabs
layout_page = dbc.Container([

    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='US Yield Curve', value='tab-1', style={'padding': '10px'}),
        dcc.Tab(label='Norwegian Yield Curve', value='tab-2', style={'padding': '10px'}),
    ], style={'marginBottom': '20px'}),  # Styling for tabs
    html.Div(id='tabs-content')
], className='', fluid=True, style={})

layout = dbc.Container([html.Div(className='beforediv'), layout_page],
    className='')

# Callback to switch between tabs
@callback(
    Output('tabs-content', 'children'),
    Input('tabs-example', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        return tab1_content
    elif tab == 'tab-2':
        return norwegian_yield_curve_layout


# Callback to update the 3D graph based on date range
@callback(
    Output('nor-yield-curve-3d', 'figure'),
    [Input('nor-date-picker-range', 'start_date'),
     Input('nor-date-picker-range', 'end_date')]
)
def update_norwegian_graph(start_date, end_date):
    # Filter data based on selected date range
    mask = (nor_yield_monthly.index >= start_date) & (nor_yield_monthly.index <= end_date)
    filtered_data = nor_yield_monthly.loc[mask]

    # Determine tick frequency
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    num_years = (end_date - start_date).days / 365.25
    
    if num_years <= 1:
        tick_freq = '3M'
    elif 1 < num_years <= 5:
        tick_freq = '6M'
    else:
        tick_freq = '1Y'

    # Create ticks
    x_ticks = pd.date_range(start=start_date, end=end_date, freq=tick_freq).to_list()
    # if nor_today not in x_ticks:
    #     x_ticks.append(nor_today)

    # Create the 3D surface plot
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


# Callback to update the graph in tab 1
@callback(
    Output('yield-curve-3df', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graph(start_date, end_date):
    # Filter the data based on selected date range
    mask = (yield_df_quarterly.index >= start_date) & (yield_df_quarterly.index <= end_date)
    filtered_data = yield_df_quarterly.loc[mask]

    # Create the 3D surface plot
    fig = go.Figure(data=[go.Surface(
        z=filtered_data.T.values,  # Use filtered quarterly data
        x=filtered_data.index, #mdates.date2num(filtered_data.index),  # Convert dates for plotting
        y=np.arange(len(maturity_labels)), 
        colorscale='GnBu',  # Color scale
        colorbar_title='Yield (%)'
    )])

    # Determine tick frequency
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    num_years = (end_date - start_date).days / 365.25
    
    if num_years <= 1:
        tick_freq = '3M'  # Quarterly
    elif 1 < num_years <= 5:
        tick_freq = '6M'  # Semi-annually
    elif 5 < num_years <= 20:
        tick_freq = '1Y'  # Annually
    else:
        tick_freq = '5Y'
    
    x_ticks = pd.date_range(start=start_date, end=end_date, freq=tick_freq).to_list()
    
    # Include today's date in x_ticks if it's not already
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
