import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta, date  # Import date
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
    'content': '#Edf_with_econ3F4'
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
    # Always load the latest CSV
    return pd.read_csv("econW_updated.csv", parse_dates=["Date"])

# Initial load
economy = load_economy_data()  # Load initial data using the update script
#print(economy.columns)
df_with_econ = pd.DataFrame()

def load_data():
    global economy, df_with_econ, latestdate, firstdate

    updateEcon.updateEcon(reload='full') 
    # 2. Reload the updated CSV
    economy = load_economy_data()
    #economy['InflationExp'] = economy['InflationExp'] / 100
    economy['unemp_rate'] = economy['unemp_rate'] / 100
    economy['TenYield'] = economy['TenYield'] / 100
    economy['Shiller_PE'] = round(economy['Shiller_PE'], 2)
    #economy['Combined Economy Score'] = round(economy['Combined Economy Score'], 2)
    #economy['Consumer Confidence'] = round(economy['ConsumerConfidence'], 2)
    economy['Close'] = round(economy['Close'], 2)
    economy['Trade Balance'] = round(economy['Trade Balance'], 0)
    economy['Trade Balance'] = economy['Trade Balance'].astype(float)*1000000
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

    df_with_econ['Interest to Income Ratio'] = ((df_with_econ['Interest Payments']) / df_with_econ['Total Revenue'])
    df_with_econ['Interest to Income Ratio'] = round(df_with_econ['Interest to Income Ratio'], 2)
    latestdate = str(pd.to_datetime(economy['Date']).dt.date.tail(1).values[0])
    firstdate = str(pd.to_datetime(economy['Date']).dt.date.head(1).values[0])
    print("Data Loaded Successfully") #Added print to confirm if data loaded.

# Load the data initially
load_data()



def create_graph(color, yaxis, title, dataframe, y, tick, starts, ends, hline1=False, textbox=False, pred=False,
                 hline0=False,
                 legend=False, YoY=False, Score=False, trade=False):
    dataframe = pd.DataFrame(dataframe).ffill().fillna(0)

    # Convert 'starts' and 'ends' to datetime.date objects if they aren't already
    if not isinstance(starts, date):
        starts = pd.to_datetime(starts).date()
    if not isinstance(ends, date):
        ends = pd.to_datetime(ends).date()

    # Convert the 'Date' column in the dataframe to datetime.date objects
    dataframe['Date'] = pd.to_datetime(dataframe['Date']).dt.date

    # Create a boolean mask for filtering
    mask = (dataframe['Date'] >= starts) & (dataframe['Date'] <= ends)
    dataframe = dataframe.loc[mask]

    # Create the figure
    fig = go.Figure()

    

    
    # Add trace with no fill color (black line)
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe[y],
        mode='lines',  # Just the line, no fill
        line_color='#2a3f5f', # Black line color
        line=dict(width=1),  # Line width 
        showlegend=False  # No legend needed for this trace
    ))

    fig.add_trace(go.Scatter(
        x=[dataframe['Date'].iloc[-1]],
        y=[dataframe[y].iloc[-1]],  # Use .iloc here for integer indexing
        mode='markers',
        marker=dict(color='red', size=6),
        showlegend=False
    ))

    last_y_value = dataframe[y].iloc[-1]
    last_x_value = dataframe['Date'].iloc[-1]

    # Format the y-value according to the tick argument
    if tick == "%":
        text = f"{last_x_value}: {last_y_value:.2%}"
    else:
        text = f"{last_x_value}: {last_y_value:.2f}"

    last_y_value = dataframe[y].iloc[-1]
    last_x_value = dataframe['Date'].iloc[-1]

    if tick == "%":
        formatted_y = f"{last_y_value:.2%}"
    else:
        formatted_y = f"{last_y_value:.2f}"

    fig.add_annotation(
        x=1,  # Right edge of plotting area
        y=1,  # Top edge of plotting area
        xref="paper",
        yref="paper",
        text=f"{last_x_value}: {formatted_y}",
        showarrow=False,
        #font=dict(color="black"),
        xanchor="right",  # Align text to the right
        yanchor="top",  # Align text to the bottom
        bordercolor='black',
        borderwidth=0.8
    )


    # Dynamically adjust y-axis range
    y_min = dataframe[y].min()
    y_max = dataframe[y].max()

    # Add a small buffer to the top and bottom of the range
    y_range_buffer = (y_max - y_min) * 0.05  # 5% buffer
    y_min -= y_range_buffer
    y_max += y_range_buffer

    # Update layout for the figure with dynamic y-axis range
    fig.update_layout(
        yaxis_title=yaxis,
        xaxis_title='Date',
        title=title,
        title_x=0.5,
        margin={'l': 0, 'r': 35},
        font=dict(family="Abel", size=15, color=colors['text']),
        plot_bgcolor='white',  # Ensures the plot area has a white background
        paper_bgcolor='white',  # Ensures the overall background is white
        yaxis=dict(range=[y_min, y_max])  # Set dynamic y-axis range
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(tickformat=".1" + str(tick))

    # Conditional traces and other configurations
    if pred == True:
        fig.add_traces(
            list(go.Scatter(x=dataframe['Date'], y=dataframe['Forward Return'], fill='tozeroy',
                             fillcolor='skyblue').select_traces()))
        fig.add_traces(
            list(go.Scatter(x=dataframe['Date'], y=dataframe['SP Trailing 4 Weeks Return'], fill='tozeroy',
                             fillcolor='red').select_traces()))

    if hline1 == True:
        fig.add_hline(y=35, line_width=3, line_dash="dash", line_color="orange")
        fig.add_hline(y=20, line_width=3, line_dash="dash", line_color="red")
    if hline0 == True:
        fig.add_hline(y=0, line_width=3, line_dash="dash", line_color="black")

    if YoY == True:
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
            borderwidth=1)

    if textbox == True:
        fig.add_annotation(
            text='Yellow Line Recommendation: 70 % Long <br> 30% Short <br> Red Line Recommendation: Risk Neutral <br> i.e 50 % Long, 50 % Short',
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0.05,
            y=1.0,
            bordercolor='black',
            borderwidth=1)

    if Score == True:
        fig.update_layout()

    if ((legend == True) & (y == 'Preds')):
        fig['data'][0]['showlegend'] = True
        fig['data'][0]['name'] = 'Predicted Forward Return'
        fig['data'][1]['showlegend'] = True
        fig['data'][1]['name'] = 'Actual Forward Return'

    # Update final layout settings
    fig.update_layout(
        font=dict(family="Helvetica", size=15, color=COLORS['text']),
        paper_bgcolor=colors['background'],
        plot_bgcolor='white',  # Ensure plot background is white
        yaxis_gridcolor=COLORS['border'],
        xaxis_gridcolor=COLORS['border'],
        height=700
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(uirevision='constant')

    return fig

# description
descriptioneconomy = f''' An overview of the US economy. Source of data is FRED API and multpl.com.'''
cardeconomy = dbc.Container([
    html.Div(children=[html.H1("Economy", style={'margin-right': '-5px'}, className='headerfinvest'),
                       html.H1("Overview", style={'color': 'rgba(61, 181, 105)', 'margin-left': '0px'},
                               className='headerfinvest'),
                       ], className='page-intros', style={'margin': '15px', 'gap': '13px'}),
    html.Div(id="description-output", children=[descriptioneconomy], className='normal-text',
             style={'max-width': '75%', 'textAlign': 'center', 'font-size': '1,5rem'}),
    html.Br(),
    dcc.Loading(  # Wrap the output component with Loading
        id="loading",
        type="default",  # or "circle" or "dot" or "cube"
        children=html.Div(id="update-output", style={'font-size': '11', 'color': 'gray'})
    ),
    html.Br(),
    html.Button('Refresh', id='refresh-button', n_clicks=0),

    html.Hr(),

    # Add DatePickerRange and RadioItems components
    html.Div([
        # dcc.DatePickerRange(
        #     id='date-picker-range',
        #     start_date=date(2010, 1, 1),  # Default start date
        #     end_date=latestdate,
        #     display_format='YYYY-MM-DD'
        # ),
        # html.Br(),
        # html.Br(),
        dcc.RadioItems(
            id='date-range-selector',
            options=[
                {'label': 'YTD', 'value': 'ytd'},
                {'label': 'Full Range', 'value': 'full'}
            ],
            style={'textAlign':'center'},
            value='full'  # Default to YTD
        ),
        html.Br()
    ]),

    html.Br(),

    html.Div([
        dcc.Graph(
            id='ten-year-yield-graph',  # Assign an ID to each graph
            style={}, className='graph'),
        # width={'size':5, 'offset':1, 'order':1},

        # xs=6, sm=6, md=6, lg=5, xl=5

        html.Div([

            dcc.Graph(id='shiller-pe-graph', className='graph'),
        ], style={'margin': '5px'}  # className='six columns' #width={'size':5, 'offset':0, 'order':2},

        ),  # className='graph-right')
    ], className='parent-row', style={'margin': '5px'}),

    html.Div([

        dcc.Graph(id='sp500-graph', className='graph'),

        html.Div([

            dcc.Graph(
                id='inflation-graph', className='graph',
                style={'border-right': '1px rgba(1, 1, 1, 1)'})

        ], style={'margin': '5px'}  # className='six columns' #width={'size':5, 'offset':0, 'order':2},

        ),
        # width={'size':5, 'offset':1, 'order':1},

        # xs=6, sm=6, md=6, lg=5, xl=5

    ], className='parent-row', style={}),

    html.Div([
        dcc.Graph(
            id='interest-to-income-graph', className='graph'),
        # width={'size':5, 'offset':1, 'order':1},

        # xs=6, sm=6, md=6, lg=5, xl=5

        html.Div([

            dcc.Graph(
                id='money-supply-graph',
                className='graph')
        ], style={'margin': '5px'}
            # className='six columns' #width={'size':5, 'offset':0, 'order':2},

        ),  # className='graph-right')
    ], className='parent-row', style={'overflow': 'visible'}),

    html.Div([
        dcc.Graph(id='t10y2y-graph', className='graph'),
        # width={'size':5, 'offset':1, 'order':1},

        # xs=6, sm=6, md=6, lg=5, xl=5

        html.Div([

            dcc.Graph(
                id='unemployment-graph', className='graph'),
        ], style={'margin': '5px'}  # className='six columns' #width={'size':5, 'offset':0, 'order':2},

        ),  # className='graph-right')
    ], className='parent-row', style={'margin': '5px'}),

    html.Div([
        dcc.Graph(id='trade-graph', className='graph')

       
        # width={'size':5, 'offset':1, 'order':1},

        # xs=6, sm=6, md=6, lg=5, xl=5

        
    ], className='parent-row', style={'margin': '5px'}),

    # html.Div([
    #     html.H3(
    #         "Below is a combined economy score visualized, which tries to give a score for the current state of the economy. The score is created by weighing fundamental factors in the economy, like the data visualized above. The data is stationary. The weights on each indicator are "
    #         "optimized in a long-short strategy of the S&P500 SPY ETF where Sharpe Ratio is maximized. Note that this score does not take into account interactions between the factors. We use data from 1998 for this purpose because of changes in economic conditions.",
    #         className='normal-text', style={'textAlign': 'center'}),
    #     html.Hr(), ], style={'margin': '5%'}),

    # html.Div([

    #     dcc.Graph(
    #         id='combined-economy-graph', style={})  # 'height':'43vw'})
    # ], className='graph', style={'width': '80%'}

    # ),
    html.Br(),
    dcc.Interval(  # Add dcc.Interval component
        id='interval-component-economy',
        interval=3600*1000*6,  # 6 hours in milliseconds
        n_intervals=0
    )

], className='parent-container2', fluid=True, style={})

layout = dbc.Container([html.Div(className='beforediv'), cardeconomy],
                       className='')


# Define the callback to update all graphs
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
     #Output('combined-economy-graph', 'figure'), 
     Output('description-output', 'children'),
     Output('update-output', 'children')],
    [#Input('date-picker-range', 'start_date'),
     #Input('date-picker-range', 'end_date'),
     Input('date-range-selector', 'value'),
     Input('interval-component-economy', 'n_intervals'),
     Input('refresh-button', 'n_clicks')],
    prevent_initial_call=False
)
def update_all_graphs(range_selector, n_intervals, n_clicks):

    """Updates all graphs based on date range and interval."""

    global economy, df_with_econ, firstdate, latestdate, descriptioneconomy #Accessing global variables
    ctx = callback_context
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'refresh-button':
            load_data()
            # Handle refresh button click specifically if needed
    # Reload data every 6 hours
    if n_intervals > 0:
        #economy = update_dataset()
        load_data()
        print(f"Data reloaded at interval: {n_intervals}")

    if range_selector == 'ytd':
        start_date = date(datetime.now().year, 1, 1)  # Set start_date to Jan 1 of current year
        start_date_infl = date(datetime.now().year, 1, 1) 
        end_date = date.today()  # Set end_date to today
    else:
        start_date = firstdate
        start_date_infl = date(1990, 1, 1)
        end_date = latestdate
    

    # Update each graph with the selected date range
    ten_year_yield = create_graph(colors['accent'], 'Yield', '10-yr Treasury Yield %', economy, 'TenYield', tick='%',
                                  starts=start_date, ends=end_date)
    shiller_pe = create_graph(colors['accent'], 'Shiller P/E Ratio', 'Shiller P/E Ratio', economy,
                              'Shiller_PE',
                              tick=' ', starts=start_date, ends=end_date)
    sp500 = create_graph(colors['accent'], 'Price', 'S&P 500 Index', economy,
                          'Close', tick=' ', starts=start_date,
                          ends=end_date)
    inflation = create_graph(colors['accent'], 'Inflation YoY', 'Inflation US YoY-Change %', economy, 'CPI YoY',
                              tick='%',
                              starts=start_date_infl, ends=end_date, YoY=True)
    interest_to_income = create_graph(colors['accent'], 'Interest to Income Ratio',
                                      'Federal Interest Payments to Revenues Ratio', df_with_econ,
                                      'Interest to Income Ratio', tick='%', starts=start_date,
                                      ends=end_date)
    money_supply = create_graph(colors['accent'], 'Money Supply M2', 'Money Supply US M2', economy,
                                'm2', tick=' ', starts=start_date, ends=end_date)
    t10y2y = create_graph(colors['accent'], 'T10Y2Y', '10-y 2-y Spread', economy,
                          'T10Y2Y', tick=' ', starts=start_date, hline0=False,
                          ends=end_date)
    unemployment = create_graph(colors['accent'], 'Unemployment Rate', 'Unemployment Rate US', economy,
                                'unemp_rate', tick='%', starts=start_date,
                                ends=end_date)
    
    tradebalance = create_graph(colors['accent'], 'Trade Balance (Exports-Imports) in Trillions $, Monthly', 'Trade Balance US in Trillions $, Monthly)', economy,
                                'Trade Balance', tick=' ', starts=start_date,
                                ends=end_date, trade=True)
    
    # combined_economy = create_graph(colors['accent'], 'Score', 'Combined Economy Score', economy,
    #                                 'Combined Economy Score',
    #                                 tick=' ', starts=start_date, ends=end_date, hline1=True,
    #                                 textbox=True, Score=True)

    #Update the  description with the latest date
    latestdate = str(pd.to_datetime(economy['Date']).dt.date.tail(1).values[0])
    descriptioneconomy = f''' An overview of the US economy. Source of data is FRED API and multpl.com.'''

    # Return all figures and the update message
    return (ten_year_yield, shiller_pe, sp500, inflation, interest_to_income, money_supply,
            t10y2y, unemployment, tradebalance, descriptioneconomy,
            f"Last check for new updates: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" )
