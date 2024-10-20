#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
import dash
from dash import html, dcc
from update_script import update_dropbox_dataset
from dash import dcc, callback, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/economy')


colors = {
    'background': 'rgb(197, 216, 239)',
    'text': 'black',
    'accent': '#004172',
    'text-white':'white',
    'content':'#EDF3F4'
}

fonts = {
    'heading': 'Arial',
    'body': 'Arial'
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

economy = pd.read_csv('https://www.dropbox.com/scl/fi/ef4rdhx9um2qyrh86narg/econW.csv?rlkey=3f75oiakw1wn6yntyv4twj5io&st=kla013ue&dl=1')
latestdate = str(pd.to_datetime(economy['Date']).dt.date.tail(1).values[0])
#'https://www.dropbox.com/scl/fi/zwcl7yhhlnk6nqg9j16r7/econW.csv?rlkey=1k0r4dnqxc4gmukgxphh0n591&dl=1'
economy['InflationExp'] = economy['InflationExp'] / 100
economy['unemp_rate'] = economy['unemp_rate'] / 100
economy['TenYield'] = economy['TenYield'] / 100
economy['Shiller_P/E'] = round(economy['Shiller_P/E'], 2)
economy['Combined Economy Score'] = round(economy['Combined Economy Score'], 2)
economy['Consumer Confidence'] = round(economy['ConsumerConfidence'], 2)
economy['Close'] = round(economy['Close'], 2)
from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt
# Initialize FRED API
FRED_API_KEY = '29f9bb6865c0b3be320b44a846d539ea'
fred = Fred(FRED_API_KEY)
# Retrieve data from FRED
# Series ID for Interest Payments: FGEXPND
# Series ID for Government Receipts: FGRECPT
interest_payments = fred.get_series('A091RC1Q027SBEA')
government_revenue = fred.get_series('FGRECPT')
#interesttogdp = fred.get_series('FYOIGDA188S')
#tenyear = fred.get_series('DGS10').resample('AS-JAN').first()

# Convert the data into a DataFrame for easier handling
interest_df = pd.DataFrame(interest_payments, columns=['Interest Payments'])
revenue_df = pd.DataFrame(government_revenue, columns=['Total Revenue'])
#interesttogdp_df = pd.DataFrame(interesttogdp, columns=['Interest to GDP'])
#tenyear_df = pd.DataFrame(tenyear, columns=['10-year'])
# Merge the two datasets on the date index
df = pd.merge(interest_df, revenue_df, left_index=True, right_index=True)
#df = pd.merge(df, interesttogdp_df,  left_index=True, right_index=True)
#df = pd.merge(df, tenyear_df,  left_index=True, right_index=True)
# Convert the index to a datetime index and extract the year
df.index = pd.to_datetime(df.index)
df.reset_index(inplace=True)
df.rename(columns={'index':'Date'}, inplace=True)

# Calculate the interest-to-income ratio
df['Interest to Income Ratio'] = ((df['Interest Payments'])/df['Total Revenue'])
df['Interest to Income Ratio'] = round(df['Interest to Income Ratio'] , 2)
# Plot the data
#plt.figure(figsize=(10, 6))
#plt.plot(df.index, df['Interest to Income Ratio'], label='Interest to Income Ratio', color='b')

# Add titles and labels
#plt.title('U.S. Government Interest-to-Income Ratio Over Time', fontsize=14)
#plt.xlabel('Year', fontsize=12)
#plt.ylabel('Interest to Income Ratio (%)', fontsize=12)


def create_graph(color, yaxis, title, dataframe, y, tick, starts, ends, hline1=False, textbox=False, pred=False,
                 legend=False, YoY=False, Score=False):
    dataframe = dataframe.ffill().fillna(0)
    mask = (dataframe['Date'] > starts) & (dataframe['Date'] <= ends)
    dataframe = dataframe.loc[mask]
    fig = px.line(dataframe, x='Date', y=y, color_discrete_sequence=[color])
    fig.update_layout(
        yaxis_title=yaxis,
        xaxis_title='Date',
        title=title,
        title_x=0.5,
        margin={
            'l': 0,
            'r': 35,
        },
        font=dict(family="Abel", size=15, color=colors['text']))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(tickformat=".1" + str(tick))
    init = 1
    fig.layout.plot_bgcolor = 'white'
    # fig.update_traces(line_color=color, line_width=1)
    if pred == True:
        fig.update_traces(line_color='orange', line_width=2)
        fig.add_traces(
            list(px.line(dataframe, x='Date', y='Forward Return', color_discrete_sequence=["skyblue"]).select_traces()))
        fig.add_traces(
            list(px.line(dataframe, x='Date', y='SP Trailing 4 Weeks Return',
                         color_discrete_sequence=["red"]).select_traces()))
        # fig.update_traces(line_width=1)
    if hline1 == True:
        fig.add_hline(y=35, line_width=3, line_dash="dash", line_color="orange")
        fig.add_hline(y=20, line_width=3, line_dash="dash", line_color="red")

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
    else:
        next
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
        #for data in fig['data']:
        #    data.hovertemplate = '%{y:.2f}<extra></extra>'
        fig.update_layout(
        #height=600

        )

    # if legend == True:
    #    fig['data'][0]['showlegend']=True
    #    fig['data'][0]['name']=y
    if ((legend == True) & (y == 'Preds')):
        fig['data'][0]['showlegend'] = True
        fig['data'][0]['name'] = 'Predicted Forward Return'
        fig['data'][1]['showlegend'] = True
        fig['data'][1]['name'] = 'Actual Forward Return'

    fig.update_layout(font=dict(family="Arial", size=15, color=COLORS['text']), # changed font to Arial
        paper_bgcolor=colors['background'], # using the background color
        plot_bgcolor='white', # using the content color for plot background
        #yaxis_tickformat=".1%", # formatting y-axis ticks
        yaxis_gridcolor=COLORS['border'], # using border color for y-grid
        xaxis_gridcolor=COLORS['border'], # using border color for x-grid
        height=700
    )

    fig.update_xaxes(showgrid=True) # show x grid for better readability
    fig.update_yaxes(showgrid=True)
    fig.update_layout(uirevision='constant')

    return fig



descriptioneconomy = f''' An overview of the US economy. Source of data is FRED API and multpl.com. Latest update: {latestdate}'''
cardeconomy = dbc.Container([
                    html.Div(children=[html.H1("Economy", style={}, className='headerfinvest'),
                                        html.H1("Overview", style={'color':'rgba(61, 181, 105)'}, className='headerfinvest'),
                                          ], className='page-intros', style={'margin':'15px'}),
            html.Div(children=[descriptioneconomy], className='normal-text', style={'max-width':'75%', 'textAlign':'center', 'font-size':'1,5rem'}),
            html.Br(),
            dcc.Loading(  # Wrap the output component with Loading
                id="loading",
                type="default",  # or "circle" or "dot" or "cube"
                children=html.Div(id="update-output", style={'font-size':'11', 'color':'gray'})
            ),
            html.Hr(),

        html.Div([
                dcc.Graph(
                    figure=create_graph(colors['accent'], 'Inflation YoY', 'Inflation US YoY-Change %', economy, 'YoY', tick='%',
                                        starts='2010-01-01', ends=str(datetime.today()), YoY=True),  className='graph',
                    style={'border-right': '1px rgba(1, 1, 1, 1)'})
                # width={'size':5, 'offset':1, 'order':1},

                # xs=6, sm=6, md=6, lg=5, xl=5

                ,
                html.Div([

                    dcc.Graph(
                        figure=create_graph(colors['accent'], 'Money Supply M2', 'Money Supply US M2', economy,
                                            'm2', tick=' ', starts='2007-01-01', ends=str(datetime.today())),
                         className='graph')
                ], style={'margin':'5px'}
                    # className='six columns' #width={'size':5, 'offset':0, 'order':2},

                ), # className='graph-right')
        ], className='parent-row', style={'overflow': 'visible'}),

        html.Div([
            dcc.Graph(
                figure=create_graph(colors['accent'], 'Yield', '10-yr Treasury Yield %', economy, 'TenYield', tick='%',
                                    starts='2010-01-01', ends=str(datetime.today())), style={},  className='graph')
                # width={'size':5, 'offset':1, 'order':1},

                # xs=6, sm=6, md=6, lg=5, xl=5

                ,
                html.Div([

                    dcc.Graph(figure=create_graph(colors['accent'], 'Shiller P/E Ratio', 'Shiller P/E Ratio', economy,
                                                  'Shiller_P/E',
                                                  tick=' ', starts='2000-01-01', ends=str(datetime.today())), className='graph'),
                ], style={'margin':'5px'}  # className='six columns' #width={'size':5, 'offset':0, 'order':2},

                ), # className='graph-right')
        ], className='parent-row', style={'margin':'5px'}),

        html.Div([
            dcc.Graph(figure=create_graph(colors['accent'], 'Confidence', 'Composite Confidence Indicator US', economy,
                                          'Consumer Confidence', tick=' ', starts='2010-01-01',
                                          ends=str(datetime.today())), className='graph'),
                # width={'size':5, 'offset':1, 'order':1},

                # xs=6, sm=6, md=6, lg=5, xl=5

                html.Div([

                    dcc.Graph(
                        figure=create_graph(colors['accent'], 'Unemployment Rate', 'Unemployment Rate US', economy,
                                            'unemp_rate', tick='%', starts='2005-01-01',
                                            ends=str(datetime.today())), className='graph'),
                ], style={'margin':'5px'}  # className='six columns' #width={'size':5, 'offset':0, 'order':2},

                ), # className='graph-right')
        ], className='parent-row', style={'margin':'5px'}),

        html.Div([
            dcc.Graph(figure=create_graph(colors['accent'], 'Price', 'S&P 500 Index', economy,
                                          'Close', tick=' ', starts='2010-01-01',
                                          ends=str(datetime.today())), className='graph'),

                html.Div([

                    dcc.Graph(
                        figure=create_graph(colors['accent'], 'Interest to Income Ratio', 'Federal Interest Payments to Revenues Ratio', df,
                                            'Interest to Income Ratio', tick='%', starts='1970-01-01',
                                            ends=str(datetime.today())), className='graph'),
                ], style={'margin':'5px'}  # className='six columns' #width={'size':5, 'offset':0, 'order':2},

                ),
                # width={'size':5, 'offset':1, 'order':1},

                # xs=6, sm=6, md=6, lg=5, xl=5

        ], className='parent-row', style={}),


        html.Div([
                html.H3(
                    "Below is a combined economy score visualized, which tries to give a score for the current state of the economy. The score is created by weighing fundamental factors in the economy, like the data visualized above. The data is stationary. The weights on each indicator are "
                    "optimized in a long-short strategy of the S&P500 SPY ETF where Sharpe Ratio is maximized. Note that this score does not take into account interactions between the factors. We use data from 1998 for this purpose because of changes in economic conditions.",
                    className='normal-text', style={'textAlign':'center'}),
                html.Hr(),], style={'margin':'5%'}),

        html.Div([

            dcc.Graph(
                figure=create_graph(colors['accent'], 'Score', 'Combined Economy Score', economy, 'Combined Economy Score',
                                    tick=' ', starts='2000-01-01', ends=str(datetime.today()), hline1=True,
                                    textbox=True, Score=True), style={})  # 'height':'43vw'})
        ], className='graph' , style={'width':'80%'}

        ),
        html.Br()
        # html.Div([
        # dbc.Card([html.Div([
        #                    html.H3("Finally we present a prediction for the next month return of the S&P500 index. The predictions are made using an LSTM Neural Network with two LSTM hidden layers. LSTM stands for Long-Short Term Memory and it fits the model "
        #                            "based on recent developments, as well as historic data to make a prediction for the future. The model is based on fundamental factors in the economy as well as price data from the S&P500 index. The model also accounts for autocorrelation"
        #                            " by looking at the data as a sequence. Based on autocorrelation analysis, we find that the data seems to be autocorrelated by a window of 1 year. We therefore choose a window of the sequence in the model as 52 weeks. The model also takes interactions between the factors into account. Measures to adjust for overfitting"
        #                            " includes both dropout and recurrent dropout layers and early stopping when the model does not improve.", style={'fontSize':'0.7em', 'font-family': ["Helvetica"], 'font-weight':'lighter'}),
        #                            ],
        #                                    style={}, className='header-items',

        # xs=12, sm=12, md=12, lg=5, xl=5)
        #   ),
        #        ])],className='header-main'),
        # html.Div([

        #           dcc.Graph(figure=create_graph('blue', 'Predicted Forward Return', 'Predictions One Month Return S&P500 LSTM Neural Network ', econW2, 'Preds', tick='%',starts='2020-01-01', ends=str(datetime.today()), hline1=False, textbox=False, pred=True, legend=True),style={})
        #       ],className='table-users' #width={'size':5, 'offset':0, 'order':2},


    # ])
], className='parent-container2', fluid=True, style={})


layout = dbc.Container([html.Div(className='beforediv'), cardeconomy],
    className='')

@callback(
    Output("update-output", "children"),
    [Input("update-button", "n_clicks")]
)
def run_update(n_clicks):
    if n_clicks is None:
        return ""
    else:
        try:
            economy = update_dropbox_dataset()
            latestdate = str(pd.to_datetime(economy['Date']).dt.date.tail(1).values[0])
            #'https://www.dropbox.com/scl/fi/zwcl7yhhlnk6nqg9j16r7/econW.csv?rlkey=1k0r4dnqxc4gmukgxphh0n591&dl=1'
            economy['InflationExp'] = economy['InflationExp'] / 100
            economy['unemp_rate'] = economy['unemp_rate'] / 100
            economy['TenYield'] = economy['TenYield'] / 100
            economy['Shiller_P/E'] = round(economy['Shiller_P/E'], 2)
            economy['Combined Economy Score'] = round(economy['Combined Economy Score'], 2)
            economy['Consumer Confidence'] = round(economy['ConsumerConfidence'], 2)
            return f"Dataset updated successfully!"
        except Exception as e:
            return f"Error updating dataset: {e}"