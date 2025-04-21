import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ClientsideFunction

dash.register_page(__name__, path='/')




layout = html.Div([
    html.Div(className='beforediv'),
    #html.Div([html.Img(src="/assets/smallelement.png", style={"pointer-events": "none", 'margin-right':'auto', 'width':'30%','height':'30%'}), html.H1('Welcome', style={'color':'black', 'margin':'auto'}, className='headerfinvest fadeinelement'),
    #          html.Img(src="/assets/bigelement.png", style={"pointer-events": "none", 'margin-left':'auto', 'width':'30%','height':'30%', 'margin-top':'10%'})], style={}, className='homewelcome'),
    #html.Img(src=('assets/littlethingbefore.jpg'),  style={'border-radius': '100px', 'width': '6%', 'margin':'15px'})], className='parent-row'),

    html.Div([
                html.Div([
                    html.Div([html.A(html.Span("Finvest"), style={'display':'inline-block', 'font-size':'2rem','text-align':'center'}, className='headers', href="/Finvest")], style={'text-align':'center'}),
                    dcc.Markdown("Create your own investment strategy based on your preferences")

                ], className='page-intro', style={'background-color': '#F9F9F9', 'border-radius': '8px', 'padding': '10px', 'box-shadow': '0px 4px 6px rgba(0, 0, 0, 0.1)'}),

                html.Div([
                    html.Div([html.A(html.Span("US Economy"), style={'display':'inline-block', 'font-size':'2rem','text-align':'center'}, className='headers', href="/economy")], style={'text-align':'center'}),
                    dcc.Markdown("Key data representing the state of the US economy")

                ], className='page-intro', style={'background-color': '#F9F9F9', 'border-radius': '8px', 'padding': '10px', 'box-shadow': '0px 4px 6px rgba(0, 0, 0, 0.1)'}),

                html.Div([
                    html.Div([html.A([html.Span("Yield Curves")], style={'display':'inline-block', 'font-size':'2rem', 'width':'75%'}, className='headers', href="/yield_curves"),], style={'text-align':'center'}) ,
                    dcc.Markdown("Historical yield curves of US and Norwegian government bonds")
                ], className='page-intro',
                style={'background-color': '#F9F9F9', 'border-radius': '8px', 'padding': '10px', 'box-shadow': '0px 4px 6px rgba(0, 0, 0, 0.1)'}),

                html.Div([
                    html.Div([html.A([html.Span("Trade War")], style={'display':'inline-block', 'font-size':'2rem', 'width':'75%'}, className='headers', href="/trade_war"),], style={'text-align':'center'}) ,
                    dcc.Markdown("Overview of the ongoing trade war")
                ], className='page-intro',
                style={'background-color': '#F9F9F9', 'border-radius': '8px', 'padding': '10px', 'box-shadow': '0px 4px 6px rgba(0, 0, 0, 0.1)'}),
                


                #html.Br(),
               ], className='page-intros fadeinelement', style={}),

    html.Br(),
    html.Div([dcc.Markdown(''' Note for this version: If you're opening this web page on mobile, please use the horizontal view ''', style={'font-size':'0.8rem', 'textAlign':'center', 'font-weight':'bold'}, className='notetext')]),
html.Br(),
], className='page-intros',
style={'background-image': 'url("/assets/pretty3.JPG")',
    'background-size': 'cover',
    'background-position': 'center',
    'min-height': '100vh',
    'position': 'relative'},
)
#, className='homecard fadeinelement',