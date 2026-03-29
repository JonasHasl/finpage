import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta, date
import dash_bootstrap_components as dbc
import dash
from dash import html, dcc
from dash import callback_context
from dash import dcc, callback, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from fredapi import Fred
import updateEcon


dash.register_page(__name__, path='/economy')


colors = {
    'background': 'rgb(240,241,245)',
    'text': 'black',
    'accent': '#004172',
    'text-white': 'white',
    'content': '#Edf3F4'  # FIX: typo corrected from '#Edf_with_econ3F4'
}

fonts = {
    'heading': 'Helvetica',
    'body': 'Helvetica'
}

COLORS = {
    'background': '#f4f4f4',
    'banner': '#0a213b',
    'banner2': '#1e3a5a',
    'content': '#859db3',
    'text': '#859db3',
    'accent': '#004172',
    'border': '#bed6eb',
    'header': '#7a7a7a',
    'element': '#1f8c44',
    'text-white': 'white',
}


def load_economy_data():
    return pd.read_csv("econW_updated.csv", parse_dates=["Date"])


# Initial load
economy = load_economy_data()
df_with_econ = pd.DataFrame()

# FIX 1: Initialise as date objects (not strings) so the YTD mask comparison
# works correctly from the very first callback before load_data() runs.
latestdate = date.today()
firstdate = date(2000, 1, 1)


def load_data():
    global economy, df_with_econ, latestdate, firstdate

    updateEcon.updateEcon(reload='incremental')
    economy = load_economy_data()

    economy['unemp_rate'] = economy['unemp_rate'] / 100
    economy['TenYield'] = economy['TenYield'] / 100
    economy['Shiller_PE'] = round(economy['Shiller_PE'], 2)
    economy['Close'] = round(economy['Close'], 2)
    economy['Trade Balance'] = round(economy['Trade Balance'], 0)
    economy['Trade Balance'] = economy['Trade Balance'].astype(float) * 1000000
    economy['Trade Balance'] = economy['Trade Balance'] / 1e12

    FRED_API_KEY = '29f9bb6865c0b3be320b44a846d539ea'
    fred = Fred(FRED_API_KEY)

    interest_payments = fred.get_series('A091RC1Q027SBEA')
    government_revenue = fred.get_series('FGRECPT')

    interest_df_with_econ = pd.DataFrame(interest_payments, columns=['Interest Payments'])
    revenue_df_with_econ = pd.DataFrame(government_revenue, columns=['Total Revenue'])

    df_with_econ = pd.merge(interest_df_with_econ, revenue_df_with_econ, left_index=True, right_index=True)
    df_with_econ.index = pd.to_datetime(df_with_econ.index)
    df_with_econ.reset_index(inplace=True)
    df_with_econ.rename(columns={'index': 'Date'}, inplace=True)

    df_with_econ['Interest to Income Ratio'] = (
        df_with_econ['Interest Payments'] / df_with_econ['Total Revenue']
    )
    df_with_econ['Interest to Income Ratio'] = round(df_with_econ['Interest to Income Ratio'], 2)

    # FIX 1 (continued): Store as datetime.date objects, NOT strings via str().
    # The original str() conversion caused a type mismatch in create_graph's mask:
    #   dataframe['Date'] (datetime.date) >= starts (str)  →  TypeError / wrong results
    latestdate = pd.to_datetime(economy['Date']).dt.date.iloc[-1]
    firstdate = pd.to_datetime(economy['Date']).dt.date.iloc[0]
    print("Data Loaded Successfully")


# Load the data initially
load_data()


def create_graph(color, yaxis, title, dataframe, y, tick, starts, ends,
                 hline1=False, textbox=False, pred=False, hline0=False,
                 legend=False, YoY=False, Score=False, trade=False):

    dataframe = pd.DataFrame(dataframe).ffill().fillna(0)

    # Guard: if the dataframe is empty or missing required columns, return blank figure
    if dataframe.empty or 'Date' not in dataframe.columns or y not in dataframe.columns:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            title_x=0.5,
            annotations=[dict(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color=COLORS['text'])
            )],
            font=dict(family="Helvetica", size=15, color=COLORS['text']),
            paper_bgcolor=colors['background'],
            plot_bgcolor='white',
            height=700
        )
        return fig

    # Convert starts/ends to datetime.date objects if they aren't already
    if not isinstance(starts, date):
        starts = pd.to_datetime(starts).date()
    if not isinstance(ends, date):
        ends = pd.to_datetime(ends).date()

    # Convert 'Date' column to datetime.date objects for comparison
    dataframe['Date'] = pd.to_datetime(dataframe['Date']).dt.date

    mask = (dataframe['Date'] >= starts) & (dataframe['Date'] <= ends)
    dataframe = dataframe.loc[mask].copy()

    # Guard: empty after date filter
    if dataframe.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            title_x=0.5,
            annotations=[dict(
                text="No data available for the selected date range",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color=COLORS['text'])
            )],
            font=dict(family="Helvetica", size=15, color=COLORS['text']),
            paper_bgcolor=colors['background'],
            plot_bgcolor='white',
            height=700
        )
        return fig

    # Reset index so iloc[-1] works correctly after .loc[] filtering
    dataframe = dataframe.reset_index(drop=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe[y],
        mode='lines',
        line_color='#2a3f5f',
        line=dict(width=1),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[dataframe['Date'].iloc[-1]],
        y=[dataframe[y].iloc[-1]],
        mode='markers',
        marker=dict(color='red', size=6),
        showlegend=False
    ))

    last_y_value = dataframe[y].iloc[-1]
    last_x_value = dataframe['Date'].iloc[-1]

    if tick == "%":
        formatted_y = f"{last_y_value:.2%}"
    else:
        formatted_y = f"{last_y_value:.2f}"

    fig.add_annotation(
        x=1,
        y=1,
        xref="paper",
        yref="paper",
        text=f"{last_x_value}: {formatted_y}",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        bordercolor='black',
        borderwidth=0.8
    )

    y_min = dataframe[y].min()
    y_max = dataframe[y].max()
    y_range_buffer = (y_max - y_min) * 0.05
    y_min -= y_range_buffer
    y_max += y_range_buffer

    fig.update_layout(
        yaxis_title=yaxis,
        xaxis_title='Date',
        title=title,
        title_x=0.5,
        margin={'l': 0, 'r': 35},
        font=dict(family="Abel", size=15, color=colors['text']),
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(range=[y_min, y_max])
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(tickformat=".1" + str(tick))

    if pred:
        fig.add_traces(
            list(go.Scatter(
                x=dataframe['Date'], y=dataframe['Forward Return'],
                fill='tozeroy', fillcolor='skyblue'
            ).select_traces()))
        fig.add_traces(
            list(go.Scatter(
                x=dataframe['Date'], y=dataframe['SP Trailing 4 Weeks Return'],
                fill='tozeroy', fillcolor='red'
            ).select_traces()))

    if hline1:
        fig.add_hline(y=35, line_width=3, line_dash="dash", line_color="orange")
        fig.add_hline(y=20, line_width=3, line_dash="dash", line_color="red")

    if hline0:
        fig.add_hline(y=0, line_width=3, line_dash="dash", line_color="black")

    if YoY:
        fig.add_hline(y=0.02, line_width=3, line_dash="dash", line_color="orange")
        fig.add_annotation(
            text='Yellow Line: FED Target Rate',
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0.05,
            y=1.0,
            bordercolor='black',
            borderwidth=1
        )

    if textbox:
        fig.add_annotation(
            text='Yellow Line Recommendation: 70 % Long <br> 30% Short <br>'
                 'Red Line Recommendation: Risk Neutral <br> i.e 50 % Long, 50 % Short',
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0.05,
            y=1.0,
            bordercolor='black',
            borderwidth=1
        )

    if legend and y == 'Preds':
        fig['data'][0]['showlegend'] = True
        fig['data'][0]['name'] = 'Predicted Forward Return'
        fig['data'][1]['showlegend'] = True
        fig['data'][1]['name'] = 'Actual Forward Return'

    fig.update_layout(
        font=dict(family="Helvetica", size=15, color=COLORS['text']),
        paper_bgcolor=colors['background'],
        plot_bgcolor='white',
        yaxis_gridcolor=COLORS['border'],
        xaxis_gridcolor=COLORS['border'],
        height=700
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(uirevision='constant')

    return fig


descriptioneconomy = ' An overview of the US economy. Source of data is FRED API and multpl.com.'

cardeconomy = dbc.Container([
    html.Div(
        children=[
            html.H1("Economy", style={'margin-right': '-5px'}, className='headerfinvest'),
            html.H1("Overview", style={'color': 'rgba(61, 181, 105)', 'margin-left': '0px'},
                    className='headerfinvest'),
        ],
        className='page-intros',
        style={'margin': '15px', 'gap': '13px'}
    ),
    html.Div(
        id="description-output",
        children=[descriptioneconomy],
        className='normal-text',
        style={'max-width': '75%', 'textAlign': 'center', 'font-size': '1.5rem'}  # FIX: was '1,5rem'
    ),
    html.Br(),
    dcc.Loading(
        id="loading",
        type="default",
        children=html.Div(id="update-output", style={'font-size': '11', 'color': 'gray'})
    ),
    html.Br(),
    html.Button('Refresh', id='refresh-button', n_clicks=0),

    html.Hr(),

    html.Div([
        dcc.RadioItems(
            id='date-range-selector',
            options=[
                {'label': 'YTD', 'value': 'ytd'},
                {'label': 'Full Range', 'value': 'full'}
            ],
            style={'textAlign': 'center'},
            value='full'
        ),
        html.Br()
    ]),

    html.Br(),

    html.Div([
        dcc.Graph(id='ten-year-yield-graph', style={}, className='graph'),
        html.Div([
            dcc.Graph(id='shiller-pe-graph', className='graph'),
        ], style={'margin': '5px'}),
    ], className='parent-row', style={'margin': '5px'}),

    html.Div([
        dcc.Graph(id='sp500-graph', className='graph'),
        html.Div([
            dcc.Graph(
                id='inflation-graph',
                className='graph',
                style={'border-right': '1px rgba(1, 1, 1, 1)'}
            )
        ], style={'margin': '5px'}),
    ], className='parent-row', style={}),

    html.Div([
        dcc.Graph(id='interest-to-income-graph', className='graph'),
        html.Div([
            dcc.Graph(id='money-supply-graph', className='graph')
        ], style={'margin': '5px'}),
    ], className='parent-row', style={'overflow': 'visible'}),

    html.Div([
        dcc.Graph(id='t10y2y-graph', className='graph'),
        html.Div([
            dcc.Graph(id='unemployment-graph', className='graph'),
        ], style={'margin': '5px'}),
    ], className='parent-row', style={'margin': '5px'}),

    html.Div([
        dcc.Graph(id='trade-graph', className='graph')
    ], className='parent-row', style={'margin': '5px'}),

    html.Br(),
    dcc.Interval(
        id='interval-component-economy',
        interval=3600 * 1000 * 6,
        n_intervals=0
    )

], className='parent-container2', fluid=True, style={})


layout = dbc.Container([html.Div(className='beforediv'), cardeconomy], className='')


@callback(
    [Output('ten-year-yield-graph', 'figure'),
     Output('shiller-pe-graph', 'figure'),
     Output('sp500-graph', 'figure'),
     Output('inflation-graph', 'figure'),
     Output('interest-to-income-graph', 'figure'),
     Output('money-supply-graph', 'figure'),
     Output('t10y2y-graph', 'figure'),
     Output('unemployment-graph', 'figure'),
     Output('trade-graph', 'figure'),
     Output('description-output', 'children'),
     Output('update-output', 'children')],
    [Input('date-range-selector', 'value'),
     Input('interval-component-economy', 'n_intervals'),
     Input('refresh-button', 'n_clicks')],
    prevent_initial_call=False
)
def update_all_graphs(range_selector, n_intervals, n_clicks):
    """Updates all graphs based on date range and interval."""

    global economy, df_with_econ, firstdate, latestdate, descriptioneconomy

    ctx = callback_context
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'refresh-button':
            load_data()

    if n_intervals > 0:
        load_data()
        print(f"Data reloaded at interval: {n_intervals}")

    # FIX 3: Defensive normalisation — convert to date objects in case
    # load_data() was skipped and the module-level strings were never replaced.
    if isinstance(firstdate, str):
        firstdate_obj = pd.to_datetime(firstdate).date()
    else:
        firstdate_obj = firstdate

    if isinstance(latestdate, str):
        latestdate_obj = pd.to_datetime(latestdate).date()
    else:
        latestdate_obj = latestdate

    if range_selector == 'ytd':
        ytd_start = date(datetime.now().year, 1, 1)

        # FIX 4: If the dataset doesn't yet reach the current calendar year
        # (common with FRED data that lags by weeks/months), fall back to
        # the start of the most recent available year to avoid empty charts.
        economy_max_date = pd.to_datetime(economy['Date']).dt.date.max()
        if economy_max_date < ytd_start:
            ytd_start = date(economy_max_date.year, 1, 1)

        start_date = ytd_start
        start_date_infl = ytd_start
        end_date = date.today()
    else:
        start_date = firstdate_obj
        start_date_infl = date(1990, 1, 1)
        end_date = latestdate_obj

    ten_year_yield = create_graph(
        colors['accent'], 'Yield', '10-yr Treasury Yield %',
        economy, 'TenYield', tick='%', starts=start_date, ends=end_date
    )
    shiller_pe = create_graph(
        colors['accent'], 'Shiller P/E Ratio', 'Shiller P/E Ratio',
        economy, 'Shiller_PE', tick=' ', starts=start_date, ends=end_date
    )
    sp500 = create_graph(
        colors['accent'], 'Price', 'S&P 500 Index',
        economy, 'Close', tick=' ', starts=start_date, ends=end_date
    )
    inflation = create_graph(
        colors['accent'], 'Inflation YoY', 'Inflation US YoY-Change %',
        economy, 'CPI YoY', tick='%', starts=start_date_infl, ends=end_date, YoY=True
    )
    interest_to_income = create_graph(
        colors['accent'], 'Interest to Income Ratio',
        'Federal Interest Payments to Revenues Ratio',
        df_with_econ, 'Interest to Income Ratio', tick='%',
        starts=start_date, ends=end_date
    )
    money_supply = create_graph(
        colors['accent'], 'Money Supply M2', 'Money Supply US M2',
        economy, 'm2', tick=' ', starts=start_date, ends=end_date
    )
    t10y2y = create_graph(
        colors['accent'], 'T10Y2Y', '10-y 2-y Spread',
        economy, 'T10Y2Y', tick=' ', starts=start_date, hline0=False, ends=end_date
    )
    unemployment = create_graph(
        colors['accent'], 'Unemployment Rate', 'Unemployment Rate US',
        economy, 'unemp_rate', tick='%', starts=start_date, ends=end_date
    )
    tradebalance = create_graph(
        colors['accent'],
        'Trade Balance (Exports-Imports) in Trillions $, Monthly',
        'Trade Balance US in Trillions $, Monthly',
        economy, 'Trade Balance', tick=' ',
        starts=start_date, ends=end_date, trade=True
    )

    descriptioneconomy = ' An overview of the US economy. Source of data is FRED API and multpl.com.'

    return (
        ten_year_yield, shiller_pe, sp500, inflation, interest_to_income,
        money_supply, t10y2y, unemployment, tradebalance,
        descriptioneconomy,
        f"Last check for new updates: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )