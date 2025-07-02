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

dash.register_page(__name__, path='/algo')


import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import datetime
import numpy as np

# from dash import DiskcacheManager, CeleryManager
# import diskcache

# # Use diskcache for async
# cache = diskcache.Cache("./cache")
# background_callback_manager = DiskcacheManager(cache)

colors = {
    'background': '#D6E4EA',
    'text': '#718BA5',
    'accent': '#004172',
    'text-white':'white',
    'content':'#EDF3F4'
}

#from sqlalchemy import create_engine

# Initialize Dash app
#app = dash.Dash(__name__)

# # SQLAlchemy engine
# engine = create_engine(
#     "mariadb+mariadbconnector://jonas:9898@192.168.0.52:3306/finwarehouse"
# )

# static_data = pd.read_csv('historical_data.csv', parse_dates=['Date'])
# static_data = static_data[static_data['Date'] <= '2024-12-31']

def calculate_dynamic_returns(weights_df, interval='1mo', start_date=None):
    symbols = weights_df['Symbol'].unique().tolist()
    dynamic_returns = []

    for quarter in weights_df[['ValidFrom', 'ValidTo']].drop_duplicates().itertuples():
        # Get price data with specified interval
        prices_raw = yf.download(
            symbols,
            start=quarter.ValidFrom if not start_date else start_date,
            end=quarter.ValidTo,
            interval=interval,
            group_by='ticker'
        )
        
        # Extract close prices correctly (handles multi-index)
        prices = pd.DataFrame()
        for symbol in symbols:
            try:
                # For multi-index structure
                prices[symbol] = prices_raw.xs('Close', level=1, axis=1)[symbol]
            except KeyError:
                print(f"Warning: No data for {symbol} in {quarter}")
                continue
                
        if prices.empty:
            continue

        prices = prices.pct_change().dropna()
        
        # Expand weights to daily dates if needed
        if interval == '1d':
            date_range = pd.date_range(start=quarter.ValidFrom, end=quarter.ValidTo, freq='B')
            weights = weights_df[weights_df['ValidFrom'] == quarter.ValidFrom]
            weights = pd.merge(pd.DataFrame({'Date': date_range}), weights, how='cross')
        else:
            weights = weights_df[weights_df['ValidFrom'] == quarter.ValidFrom]

        weighted_returns = prices[weights['Symbol']].mul(weights['Weight'].values).sum(axis=1)
        
        
        dynamic_returns.extend([{
            'Date': date,
            'Return': ret,
            'Symbol': 'Top',
            'Frequency': interval
        } for date, ret in weighted_returns.items()])
    # if interval == '1d':
    #     returnsfinal = pd.DataFrame(dynamic_returns)
    #     returnsfinal.loc[returnsfinal['Symbol'] == 'Top', 'Return'] = returnsfinal.loc[returnsfinal['Symbol'] == 'Top', 'Return'] / 100
    # else:
    #     returnsfinal = pd.DataFrame(dynamic_returns)

    returnsfinal = pd.DataFrame(dynamic_returns)

    # Convert percent to decimal if values are unusually high
    if interval == '1d' and returnsfinal['Return'].abs().mean() > 0.5:
        returnsfinal.loc[returnsfinal['Symbol'] == 'Top', 'Return'] /= 100

    return returnsfinal



# Fetch and prepare data using SQLAlchemy
def fetch_data(use_daily_ytd=False):
    # Load static data for both series
    static_top = pd.read_csv(r"algoreturnshistory.csv", parse_dates=['Date']).drop(columns=['Spalte 1'])
    static_acwi = pd.read_csv(r"acwi.csv", parse_dates=['Date'])
    static_top = static_top[static_top.Date <= '2024-12-31']
    static_acwi = static_acwi[static_acwi.Date <= '2024-12-31']
    # Calculate dynamic returns for Top portfolio
    weights_df = pd.read_csv(r"algocomposition.csv")
    if use_daily_ytd:
        # Daily returns for YTD
        dynamic_top = calculate_dynamic_returns(weights_df, interval='1d', start_date='2025-01-01')
        dynamic_acwi = fetch_acwi_returns(interval='1d', start_date='2025-01-01')
    else:
        # Monthly returns for full range
        dynamic_top = calculate_dynamic_returns(weights_df, interval='1mo')
        dynamic_acwi = fetch_acwi_returns(start_date='2025-01-01', interval='1mo')
    finalframe = pd.concat([static_top, static_acwi, dynamic_top, dynamic_acwi]).drop_duplicates(subset=['Symbol', 'Date'])
    return finalframe


def fetch_acwi_returns(start_date, interval='1mo'):
    
    acwi_prices = yf.download(
        'EUNL.F', 
        start=start_date,
        end=pd.Timestamp.today(),
        interval=interval
    )['Close']

    
    acwi_returns = acwi_prices['EUNL.F'].pct_change().dropna()
    return pd.DataFrame({
        'Date': acwi_returns.index,
        'Return': acwi_returns.values,
        'Symbol': 'ACWI'
    })


# Get USD/NOK exchange rates
def get_fx_rates():
    fxms = yf.download('NOK=X', start='2010-01-01', end=datetime.today(), interval='1mo')
    fxms = fxms['Close'].resample('MS').first().reset_index()
    fxms['NOK=X'] = fxms['NOK=X'].pct_change()

    fxme = yf.download('NOK=X', start='2010-01-01', end=datetime.today(), interval='1mo')
    fxme = fxme['Close'].resample('ME').last().reset_index()
    fxme['NOK=X'] = fxme['NOK=X'].pct_change()
    
    fx = pd.concat([fxms, fxme], axis=0).sort_values('Date')
    return fx

# Modified calculate_metrics function
# Modified calculate_metrics function
def calculate_metrics(df, fx_data=None):
    # Calculate USD returns first
    df = df.drop_duplicates(subset=['Symbol', 'Date'])
    df = df.sort_values(['Symbol', 'Date'])
    if fx_data is None:
        df['CumulativeReturn'] = df.groupby('Symbol')['Return'].transform(
            lambda x: (1 + x).cumprod() - 1
    )

    if fx_data is not None:
        # Merge FX returns
        df = df.merge(fx_data, on='Date', how='left')
        
        # Calculate NOK-adjusted returns: (1 + Return) * (1 + FX_Return) - 1
        df['NOK_Return'] = (1 + df['Return']) * (1 + df['NOK=X']) - 1
        
        # Recalculate cumulative returns with NOK-adjusted values
        df['CumulativeReturn'] = df.groupby('Symbol')['NOK_Return'].transform(
            lambda x: (1 + x).cumprod() - 1
        )
        
        # Use NOK returns for subsequent calculations
        return_col = 'NOK_Return'
    else:
        return_col = 'Return'

    # Annual returns calculation
    df['Date'] = pd.to_datetime(df['Date'])
    annual_returns = df.groupby([df['Date'].dt.year, 'Symbol'])[return_col].apply(
        lambda x: (1 + x).prod() - 1
    ).unstack()
    print(df.groupby([df['Date'].dt.year, 'Symbol'])['Return'].apply(lambda x: (1 + x).prod() - 1).unstack())
    # YTD calculation
    current_year = datetime.now().year
    ytd_data = df[df['Date'].dt.year == current_year]
    ytd_data = df[
    (df['Date'].dt.year == current_year) &
    (df['Date'] <= datetime.now())
        ]
    ytd_data = ytd_data.drop_duplicates(subset=['Symbol', 'Date'])
    ytd_returns = {
        security: (1 + grp[return_col]).prod() - 1
        for security, grp in ytd_data.groupby('Symbol')
    }
    
    # Risk metrics calculation
    df = df.drop_duplicates(subset=['Symbol', 'Date'])
    returns_pivot = df.pivot_table(
        index='Date', 
        columns='Symbol', 
        values=return_col
    ).dropna(how='any', subset=['ACWI', 'Top'])  # Remove rows with NaN in either series

    # Validate data presence
    if 'ACWI' not in returns_pivot.columns or 'Top' not in returns_pivot.columns:
        raise ValueError("Missing required series in pivot table")

    # Ensure identical time periods
    returns_pivot = returns_pivot.asfreq('M')  # Enforce monthly frequency
    returns_pivot = returns_pivot#.ffill().bfill()  # Handle any remaining gaps

    # Final NaN check
    if returns_pivot[['ACWI', 'Top']].isna().any().any():
        returns_pivot = returns_pivot.dropna(subset=['ACWI', 'Top'])
    model = LinearRegression().fit(returns_pivot[['ACWI']].values, returns_pivot[['Top']].values)
    
    metrics = {
        'annual_returns': annual_returns,
        'average_annual_returns': annual_returns.mean(),
        'ytd_returns': ytd_returns,
        'beta': model.coef_[0][0],
        'alpha': model.intercept_[0]
    }
    
    print("Duplicate dates per symbol in YTD:")
    if 'ACWI' in df['Symbol'].unique():
        acwi_ytd = df[(df['Symbol'] == 'ACWI') & (df['Date'].dt.year == current_year)]
    print("\n--- ACWI YTD CHECK ---")
    print(acwi_ytd.tail(10))
    print("ACWI YTD duplicate dates:")
    print(acwi_ytd['Date'].duplicated().sum())
    print("ACWI YTD return stats:")
    print(acwi_ytd['Return'].describe())

    print(df[df['Date'].dt.year == current_year].groupby('Symbol')['Date'].value_counts().loc[lambda x: x > 1])


    return df, metrics, return_col

description = ''' This dashboard visualizes the performance of a stock trading algorithm based on Morningstar fundamental data. The strategy consists of 7 stocks rebalanced quarterly which are selected
based on a combined alpha score. The combined alpha score is calculated by ranking the stocks in a universe consisting of S&P 500 and Nasdaq 100 constituents. After the stocks are ranked based on each factor, each stock will receive
a combined alpha score which is based on a weighed sum of these ranks. The stocks with the highest combined alpha score are included in the strategy. The weight on each factor
are optimized to maximize the Sharpe ratio of the portfolio. The algorithm is designed to be robust and adaptive, with a focus on long-term performance. '''


# Updated app layout
layout = dbc.Container([html.Div(className='beforediv'),
    dbc.Row(dbc.Col(html.H1("Portfolio Performance",className='headerfinvest', style={'textAlign':'center'}))),
    html.Div(children=[description, html.Hr()], className='normal-text', style={'textAlign':'center', 'font-size':'1,5rem', 'margin-left':'10%', 'margin-right':'10%', 'font-weight':'bold'}),
    
    dbc.Row([
        dbc.Col(dcc.RadioItems(
            id='currency-selector',
            options=[
                {'label': 'USD', 'value': 'USD'},
                {'label': 'NOK', 'value': 'NOK'}
            ],
            value='USD',
            labelStyle={'display': 'inline-block', 'margin-right': '10px'},
            style={'textAlign': 'center'}
        ), width=3),
        dbc.Col(dcc.RadioItems(
            id='range-selector',
            options=[
                {'label': 'Full Range', 'value': 'full'},
                {'label': 'YTD', 'value': 'ytd'}
            ],
            value='full',
            labelStyle={'display': 'inline-block', 'margin-right': '10px'},
            style={'textAlign': 'center'}
        ), width=3)
    ],style={'textAlign': 'center'}),
    html.Br(),
    dbc.Row([  # Graph on top
        dbc.Col(dcc.Graph(id='cumulative-returns-chart'), width=12)
    ]),
    dbc.Row([  # New drawdown graph
        dbc.Col(dcc.Graph(id='drawdown-chart'), width=12)
    ]),
    html.Br(),
    
    html.Div([
    dbc.Row([html.Div([  # Portfolio Total Return and Std Dev
        dbc.Col(html.Div(id='portfolio-total-card'), width=3),
        dbc.Col(html.Div(id='portfolio-std-card'), width=3)]
    )]),
    
    dbc.Row([  # Portfolio and ACWI YTD Total Return (color-coded)
        dbc.Col(html.Div(id='portfolio-annual-returns-card'), width=6),
        dbc.Col(html.Div(id='acwi-annual-returns-card'), width=6)
    ]),
    
    dbc.Row([html.Div([
    dbc.Col(html.Div(id='beta-card'), width=6),
    dbc.Col(html.Div(id='alpha-card'), width=6)
    ])]),
    
    dbc.Row([  # ACWI Total Return and Std Dev
        dbc.Col(html.Div(id='acwi-total-card'), width=6),
        dbc.Col(html.Div(id='acwi-std-card'), width=6)
    ]),
    
    dbc.Row([  # Portfolio and ACWI YTD Total Return (color-coded)
        dbc.Col(html.Div(id='portfolio-ytd-card'), width=6),
        dbc.Col(html.Div(id='acwi-ytd-card'), width=6)
    ])
    
    
    
    ],
    className='my-card-container', style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap', 'gap': '20px'}),

    html.Br(),
    
    dbc.Row([  # Portfolio composition table
        dbc.Col(html.Div(id='portfolio-table'), width=12)
    ]),
    dbc.Row([
    dbc.Col(
        dbc.Button(
            "Show Stocks in Latest Composition Cumulative Returns",
            id="show-comps-btn",
            color="primary",
            className="buttonDefinitions",
            style={'width': 'auto', 'height': 'auto', 'padding':'20px', 'margin': '10px'}  # Full width button
        ),
        width=12
    )
        ], style={'textAlign': 'center'}, className='my-button-container'), 
        dbc.Row([
    dbc.Col(
            html.Div(  # Add this wrapper div
                dcc.Graph(id='component-cumulative-chart'),
                id="graph-container",
                style={'display': 'none'}  # Initially hidden
            ),
            width=12
        )
    ]),
    html.Br(),



    
    dbc.Row([  # Upload button
        dbc.Col(dcc.Upload(
            id='upload-weights',
            children=html.Div(['Drag & Drop or ', html.A('Select Excel File')]),
            style={'borderStyle': 'dashed', 'padding': '20px'}
        ), width=12)
    ])
])

from dash import Input, Output, State


@callback(
    Output('component-cumulative-chart', 'figure'),
    Output('graph-container', 'style'),  # Controls visibility
    Input('show-comps-btn', 'n_clicks'),
    State('range-selector', 'value'),
    prevent_initial_call=True
)
def update_component_cumulative_chart(n_clicks, date_range):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # [Rest of your existing code here...]
    # Get latest portfolio composition
    weights_df = pd.read_csv(r"algocomposition.csv")
    latest_date = weights_df['ValidFrom'].max()
    symbols_stocks = pd.DataFrame()
    symbols_stocks = weights_df[weights_df['ValidFrom'] == latest_date]['Symbol'].unique().tolist()

    # Set date range
    start_date = datetime(datetime.now().year, 1, 1)
    end_date = datetime.now()

    # Download all symbols at once
    try:
        prices_raw = yf.download(
            symbols_stocks,
            start=start_date,
            end=end_date,
            interval='1d',
            group_by='ticker'
        )
    except Exception as e:
        print(f"Error downloading data: {e}")
        return go.Figure()

    # Process prices using your existing pattern
    cumulative_returns_stocks = {}
    returns = None
    print(f'Number of symbols: {len(symbols_stocks)}')
    for symbol in symbols_stocks:
        try:
            closes = prices_raw.xs('Close', level=1, axis=1)[symbol].dropna()
            if not closes.empty:
                returns = closes.pct_change().fillna(0)  # Set first return to 0
                cumulative = (1 + returns).cumprod() - 1
                cumulative_returns_stocks[symbol] = cumulative
        except KeyError:
            print(f"No data for {symbol}")
            continue

    # Get complete date index from prices_raw
    # Create a complete business day index
    date_index = pd.date_range(start=start_date, end=end_date, freq='B')

    # Build DataFrame with forward-filling
    df_graph = pd.DataFrame(index=date_index)
    for symbol in symbols_stocks:
        if symbol in cumulative_returns_stocks:
            # Forward-fill then back-fill missing values
            s = cumulative_returns_stocks[symbol].reindex(date_index)
            df_graph[symbol] = s.ffill()


    # Clean up NaNs using your existing pattern
    #df = df.dropna(how='all')#.ffill().bfill()
    df_graph.index = pd.to_datetime(df_graph.index)

    # Use your existing color scheme
    top_color = '#1A5276'      # Deep Navy
    acwi_color = '#3498DB'     # Bright Blue
    #color_sequence = [top_color, acwi_color, '#27AE60', '#8E44AD', '#F39C12']

    fig = go.Figure()
    for _, symbol in enumerate(df_graph.columns):
        fig.add_trace(
            go.Scatter(
                x=df_graph.index,
                y=df_graph[symbol],
                mode='lines',
                name=symbol,
                line=dict(width=1.5)
            )
        )

    fig.update_layout(
        title='Cumulative YTD Returns (Stock Components)',
        yaxis=dict(
            tickformat='.1%',
            title='Cumulative Return',
            rangemode='tozero'
        ),
        xaxis=dict(
            title='Date',
            showgrid=False
        ),
        height=400,
        margin=dict(l=50, r=50, t=60, b=50),
        hovermode='x unified',
        plot_bgcolor='white',
        font=dict(
            family='Helvetica Neue, Arial, sans-serif',
            color='#333'
        ),
        title_font=dict(
            size=20,
            color=top_color
        )
    )

    average_return = df_graph.iloc[-1,:].mean()

    fig.add_hline(
        y=average_return,
        line_dash="dash",
        line_color="firebrick",
        annotation_text=f"Mean Return Stocks in Graph: {average_return:.2%}",
        annotation_position="top right"
)


    return fig, {'display': 'block'}





@callback(
    Output('currency-selector', 'options'),
    Output('range-selector', 'options'),
    Input('currency-selector', 'value'),
    Input('range-selector', 'value')
)
def update_selector_options(currency, date_range):
    # Currency selector options
    currency_options = [
        {'label': 'USD', 'value': 'USD'},
        {'label': 'NOK', 'value': 'NOK'}
    ]
    # Range selector options
    range_options = [
        {'label': 'Full Range', 'value': 'full'},
        {'label': 'YTD', 'value': 'ytd'}
    ]

    # Disable NOK in currency selector if YTD is selected
    if date_range == 'ytd':
        currency_options[1]['disabled'] = True
        # Force currency to USD if YTD is selected
        currency = 'USD'
    else:
        currency_options[1]['disabled'] = False

    # Disable YTD in range selector if NOK is selected
    if currency == 'NOK':
        range_options[1]['disabled'] = True
        # Force range to full if NOK is selected
        date_range = 'full'
    else:
        range_options[1]['disabled'] = False

    return currency_options, range_options



# Modified callback
@callback(
    [Output('cumulative-returns-chart', 'figure'),
     Output('drawdown-chart', 'figure'),
     Output('portfolio-total-card', 'children'),
     Output('portfolio-std-card', 'children'),
     Output('acwi-total-card', 'children'),
     Output('acwi-std-card', 'children'),
     Output('portfolio-ytd-card', 'children'),
     Output('acwi-ytd-card', 'children'),
     Output('portfolio-table', 'children'),
     Output('alpha-card', 'children'),
     Output('beta-card', 'children'),
     Output('portfolio-annual-returns-card', 'children'),
     Output('acwi-annual-returns-card', 'children')],
    [Input('currency-selector', 'value'),
     Input('range-selector', 'value')]
)
def update_dashboard(currency, date_range):
    current_year = datetime.now().year
    use_daily = date_range == 'ytd'
    df = fetch_data(use_daily_ytd=use_daily)
    
    ytd_df = fetch_data(use_daily_ytd=True)  # Daily data for current year
    ytd_df = ytd_df[ytd_df['Date'].dt.year == current_year]
    
    if use_daily:
        # If the Top series returns are in percent, divide by 100
        #df.loc[df['Symbol'] == 'Top', 'Return'] = df.loc[df['Symbol'] == 'Top', 'Return'] / 100
        df = df[df['Date'] <= datetime.now()]

    fx_data = get_fx_rates() if (currency == 'NOK' and date_range != 'ytd') else None

    df, metrics, return_col = calculate_metrics(df, fx_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date'])
    df = df.drop_duplicates(subset=['Symbol', 'Date'])
    # Calculate metrics using YTD-specific data
    ytd_df, ytd_metrics, return_col = calculate_metrics(ytd_df, fx_data)
    # Date range filtering for graph only
    if date_range == 'ytd':
        graph_df = ytd_df
    else:
        graph_df = df

    # Recalculate cumulative returns for graph only
    graph_df = graph_df.reset_index(drop=True)

    graph_df['FilteredCumulative'] = graph_df.groupby('Symbol', group_keys=False).apply(
        lambda x: (1 + x[return_col]).cumprod() - 1
    )
    
    

    # for security, grp in graph_df.groupby('Symbol'):
    #     print(security)

    # Ensure dates are sorted
    graph_df = graph_df.sort_values(['Symbol', 'Date'])
    graph_df.loc[graph_df['Symbol'] == 'Top', 'Symbol'] = 'Strategy'
    if date_range != 'ytd':
        print(graph_df.tail(20))
    # Create figure with filtered data
    top_color = '#1A5276'      # Deep Navy
    acwi_color = '#3498DB'     # Bright Blue
    max_drawdown_color = '#FF0000'  # Red

    # Update the fig creation
    fig = {
        'data': [
            {
                'x': grp['Date'],
                'y': grp['FilteredCumulative'],
                'name': security,
                'mode': 'lines',
                'line': {'color': top_color if security == 'Strategy' else acwi_color}
            } for security, grp in graph_df.groupby('Symbol')
        ],
        'layout': {
            'title': 'Cumulative Returns Comparison',
            'yaxis': {
                'tickformat': '.3%',
                'rangemode': 'tozero'
            }
        }
    }


    def calculate_drawdowns(group):
        group['Peak'] = group['CumulativeReturn'].cummax()
        group['Drawdown'] = (group['CumulativeReturn'] - group['Peak']) / (1 + group['Peak'])
        return group

    graph_df = graph_df.groupby('Symbol', group_keys=False).apply(calculate_drawdowns)
    graph_df.loc[graph_df['Symbol'] == 'Top', 'Symbol'] = 'Strategy'
    # Create drawdown figure
    drawdown_fig = {
    'data': [
        {
            'x': grp['Date'],
            'y': grp['Drawdown'],
            'name': security,
            'mode': 'lines',
            'line': {'color': top_color if security == 'Strategy' else acwi_color}
        } for security, grp in graph_df.groupby('Symbol')
    ],
    'layout': {
        'title': 'Portfolio Drawdowns',
        'yaxis': {'tickformat': '.0%'},
        'shapes': [{
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'x1': 1,
            'y0': graph_df[graph_df['Symbol'] == 'Strategy']['Drawdown'].min(),
            'y1': graph_df[graph_df['Symbol'] == 'Strategy']['Drawdown'].min(),
            'line': {
                'color': max_drawdown_color,
                'dash': 'dot'
            },
            'name': 'Max Drawdown (Strategy)'
        }]
        }
    }
    
    # Calculate standard deviations (using all available data)
    returns_pivot = df.pivot_table(index='Date', columns='Symbol', values='Return')
    portfolio_std = returns_pivot['Top'].std() * np.sqrt(12)  # Annualized
    acwi_std = returns_pivot['ACWI'].std() * np.sqrt(12) if 'ACWI' in returns_pivot.columns else 0

    # Total return calculations
    portfolio_total = df[df['Symbol'] == 'Top']['CumulativeReturn'].iloc[-1] if not df[df['Symbol'] == 'Top'].empty else 0
    acwi_total = df[df['Symbol'] == 'ACWI']['CumulativeReturn'].iloc[-1] if not df[df['Symbol'] == 'ACWI'].empty else 0

    annual_portfolio_returns = metrics['average_annual_returns'].get('Top', 0)
    annual_acwi_returns = metrics['average_annual_returns'].get('ACWI', 0)

    # YTD returns (always current year)
    portfolio_ytd = ytd_metrics['ytd_returns'].get('Top', 0)
    acwi_ytd = ytd_metrics['ytd_returns'].get('ACWI', 0)

    # Create cards
    portfolio_total_card = dbc.Card(
        dbc.CardBody([
            html.H5("Strategy Total Return", style={'textAlign':'center'}),
            html.P(f"{portfolio_total:.1%}", className="card-text")
        ]), style={'backgoundColor':'white'}
    )

    portfolio_std_card = dbc.Card(
        dbc.CardBody([
            html.H5("Strategy Annual Risk", style={'textAlign':'center'}),
            html.P(f"{portfolio_std:.1%}", className="card-text")
        ]), style={'backgoundColor':'white'}
    )

    portfolio_annual_returns_card = dbc.Card(
        dbc.CardBody([
            html.H5("Strategy Annual Return", style={'textAlign':'center'}),
            html.P(f"{annual_portfolio_returns:.1%}", className="card-text")
        ]), style={'backgoundColor':'white'}
    )
    
    acwi_annual_returns_card = dbc.Card(
        dbc.CardBody([
            html.H5("ACWI Annual Return", style={'textAlign':'center'}),
            html.P(f"{annual_acwi_returns:.1%}", className="card-text")
        ]), style={'backgoundColor':'white'}
    )
    
    acwi_total_card = dbc.Card(
        dbc.CardBody([
            html.H5("ACWI Total Return", style={'textAlign':'center'}),
            html.P(f"{acwi_total:.1%}", className="card-text")
        ]), style={'backgoundColor':'white'}
    )
    
    

    acwi_std_card = dbc.Card(
        dbc.CardBody([
            html.H5("ACWI Annual Risk", style={'textAlign':'center'}),
            html.P(f"{acwi_std:.1%}", className="card-text")
        ]), style={'backgoundColor':'white'}
    )
    
    portfolio_ytd_card = dbc.Card(
        dbc.CardBody([
            html.H5("Strategy YTD Return", style={'textAlign':'center'}),
            html.P(f"{portfolio_ytd:.1%}", className="card-text")
        ]), style={'backgoundColor':'white'}
    )
    
    acwi_ytd_card = dbc.Card(
        dbc.CardBody([
            html.H5("ACWI YTD Return", style={'textAlign':'center'}),
            html.P(f"{acwi_ytd:.1%}", className="card-text")
        ]), style={'backgoundColor':'white'}
    )
    
    # Risk metrics cards
    beta_card = dbc.Card(
        dbc.CardBody([
            html.H5("Strategy Beta", style={'textAlign':'center'}),
            html.P(f"{metrics['beta']:.2f}", className="card-text")
        ]), style={'backgoundColor':'white'}
    )
    
    alpha_card = dbc.Card(
        dbc.CardBody([
            html.H5("Strategy Alpha", style={'textAlign':'center'}),
            html.P(f"{(1+metrics['alpha'])**12-1:.2%}", className="card-text")
        ]), style={'backgoundColor':'white'}
    )
    
    # Portfolio composition table
    weights_df = pd.read_csv(r"algocomposition.csv")
    latest_composition = weights_df[weights_df['ValidFrom'] == weights_df['ValidFrom'].max()]
    latest_composition['Weight'] = latest_composition['Weight'].round(2)
    # Update your DataTable definition in your layout or callback

    table = dash.dash_table.DataTable(
        data=latest_composition.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in ['Symbol', 'Weight', 'ValidFrom', 'ValidTo']],
        editable=False,
        filter_action="none",
        # sort_action="native",
        # sort_mode="multi",
        column_selectable="single",
        row_selectable=False,
        row_deletable=False,
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
                'column_id': 'Symbol'
            },
            'textAlign': 'left',
            'backgroundColor': '#9fa4d8'
        }],

        style_header={
            'backgroundColor': colors['accent'],
            'fontWeight': 'bold',
            'color': 'white',
            'whiteSpace': 'normal',
            'border': '2px solid black',
            'font-family': ['Arial'],
            'font-size':'15px'
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
                    'column_id': 'Symbol'
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
        }
    )


    return (
        fig,
        drawdown_fig,
        portfolio_total_card,
        portfolio_std_card,
        acwi_total_card,
        acwi_std_card,
        portfolio_ytd_card,
        acwi_ytd_card,
        table,
        alpha_card,
        beta_card,
        portfolio_annual_returns_card,
        acwi_annual_returns_card
    )


# if __name__ == '__main__':
#     app.run(debug=False)
