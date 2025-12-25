from dash import html, dcc
import dash
import dash_bootstrap_components as dbc
import re
from dash.dependencies import Input, Output, State
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = dash.Dash(__name__, use_pages=True,
                external_stylesheets=['custom.css', 'missing.css',
    'https://use.fontawesome.com/releases/v5.8.1/css/all.css'],
                meta_tags=[{'name' :'viewport', 'content':'width=device-width, initial-scale=1, height=device-height'}], suppress_callback_exceptions=True, prevent_initial_callbacks="initial_duplicate")


server = app.server

colors = {
    'background': '#D6E4EA',
    'text': '#718BA5',
    'accent': '#004172',
    'text-white':'white',
    'header': '#004172'
}

navbar_style = {
    "border": "none",
    "color": '#eee',
    'text-transform': 'none',
    'font-size':'24px',
    'textAlign':'right'
}

data_url_triangle = """
data:image/svg+xml;utf8,<svg width="12" height="10" xmlns="http://www.w3.org/2000/svg"><polygon points="0,0 12,0 6,10" style="fill:white"/></svg>
"""

data_url_triangle_right = """
data:image/svg+xml;utf8,<svg width="10" height="12" xmlns="http://www.w3.org/2000/svg"><polygon points="0,0 0,12 10,6" style="fill:white"/></svg>
"""

header_banner = dbc.Navbar(
    [

        dbc.Nav(
            [html.A("", href="/", style={'font-size':'24px', 'font-weight':'lighter', 'color': '#7a7a7a', 'margin-right': 'auto', 'margin-left':'20px'} ),
                html.Div([
                dbc.NavItem(dbc.NavLink("Home", href="/", style={
                                                            "border": "none",
                                                            "color": '#7a7a7a',
                                                            'text-transform': 'none',
                                                            'font-size':'24px',
                                                            'textAlign':'right'

                                                        },), className='menuitemtop'),

                dbc.NavItem(dbc.NavLink("Finvest", href="/Finvest", style={
                        "border": "none",
                        "color": '#7a7a7a',
                        'text-transform': 'none',
                        'font-size':'24px',
                        'textAlign':'right'

                    }), className='menuitemtop'),
                
                
                dbc.NavItem(dbc.NavLink("Economy Overview", href="/economy", style={
                                                            "border": "none",
                                                            "color": '#7a7a7a',
                                                            'text-transform': 'none',
                                                            'font-size':'24px',
                                                            'textAlign':'right'

                                                        }), className='menuitemtop'),
                dbc.NavItem(dbc.NavLink("Yield Curve", href="/yield_curves", style={
                                                            "border": "none",
                                                            'font-size':'24px',
                                                            "color": '#7a7a7a',
                                                            'text-transform': 'none',
                                                            'textAlign':'right'

                                                        },), className='menuitemtop'),
                dbc.NavItem(dbc.NavLink("Algorithm", href="/portfolio-daily", style={
                                                            "border": "none",
                                                            'font-size':'24px',
                                                            "color": '#7a7a7a',
                                                            'text-transform': 'none',
                                                            'textAlign':'right'

                                                        },), className='menuitemtop'),
                # dbc.NavItem(dbc.NavLink("Trade War", href="/trade_war", style={
                #                                             "border": "none",
                #                                             'font-size':'24px',
                #                                             "color": '#7a7a7a',
                #                                             'text-transform': 'none',
                #                                             'textAlign':'right'

                #                                         },), className='menuitemtop'),
                    ], id='output_div2', className='banner-row', style={'display':''}),
            #html.Button(' ', id='show_hide_button2', className='buttonMenu', style={'display':''}),
            ],
            style={'padding-right':'20px'},
            className='banner-row',
            navbar=True
        ),
    ],
    sticky="top",
    className='headerbanner',
style={'position':'absolute', 'z-index':'10003', '-webkit-font-smoothing': 'antialiased', 'width':'100%'},

)




menu = html.Div([
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'Link 1', 'value': 'link1'},
            {'label': 'Link 2', 'value': 'link2'},
            {'label': 'Link 3', 'value': 'link3'}
        ],
        placeholder='...',
        searchable=False,
        clearable=False,
        style={
            'position': 'absolute',
            'top': '10px',
            'right': '10px',
            'width': '150px',
            'border': 'none',
            'background-color': 'transparent',
            'font-size': '1.2em',
'           dropdownIndicator': {
                'color': 'white'}
        },
        #indicator_style={
        #    'display': 'none'
        #}


    )
])

# secondary_menu = html.Div(
#     [
#         dbc.Nav(
#             [
#                 dbc.NavLink(html.Span(["Instruksjoner"],style={'color':'white'}) , href="/Instruksjoner", style={'color':'white'}),
#                 dbc.NavLink("Hvordan handle kurtasjefritt", href="/kurtasjefritt", style={'color':'white'}),
#                 dbc.NavLink("Hvorfor denne tjenesten?", href="/tjenesten", style={'color':'white'})
#             ],
#             style={'justify-content':'center'}, # A darker shade of blue
#             className="banner-row",
#             navbar=True
#         )
#     ],
#     className='secondary-menu-container',
#     style={'display': 'none'},  # This hides the menu initially
#     id='secondary-menu-container'
# )


# finvest_menu = html.Div([
#     dbc.Nav([
#         dbc.NavLink(
#             html.Span("S&P500 & NASDAQ 100", style={'font-size':'1rem','text-align':'center', 'color':'black', 'padding-bottom':'2px'}, className='headers_finvest_menu'),
#             className="us-tab",
#             href="Finvest-US"
#         ),
#         dbc.NavLink(
#             html.Span('Scandinavian Stocks', style={'font-size':'1rem','text-align':'center', 'color':'black', 'padding-bottom':'2px'}, className='headers_finvest_menu'),
#             className='scan-tab',
#             href="/Finvest-SC"
#         )
#     ],
#     style={'justify-content': 'center', 'gap':'20px'},
#     className="tabs-country",
#     navbar=True)
# ], className='finvest-menu-container', style={'display': 'none'}, id='finvest-menu-container')





contact = html.Div([

    html.H3("Contact Information"),
    html.A(html.Span("jonas_fbh@hotmail.com", style={'display':'inline-block', 'font-size':'1rem','text-align':'center', 'color':'black', 'padding-bottom':'2px'}), href="mailto:jonas_fbh@hotmail.com"),
    html.Br(),
    html.A(html.Span("https://www.linkedin.com/in/jonashasl", style={'display':'inline-block', 'font-size':'1rem','text-align':'center', 'color':'black', 'padding-bottom':'2px'}), href="https://www.linkedin.com/in/jonashasl"),
    html.Br(),
    html.A(html.Span("+47 45101329", style={'display':'inline-block', 'font-size':'1rem','text-align':'center', 'color':'black', 'padding-bottom':'2px'}),href="tel:+4745101329")],
    style={'color':'black', 'font-weight':'normal', 'textAlign':'center'}, className='footer-left')

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
    header_banner]),
    dash.page_container
    #html.Hr(style={'margin-block-start': '0em'}),
    #html.Div([contact], className='footer')
], style={'margin-top':'0'},
) #className='parent-container')

# @app.callback(
#     Output(component_id='output_div2', component_property='style'),
#     Output('output_div2', 'className'),
#     [Input(component_id='show_hide_button2', component_property='n_clicks')],
#     [State(component_id='output_div2', component_property='style')]
# )

# def show_hide_element(n_clicks, style):
#     if n_clicks is None:
#         style['display'] = ''
#         return style, 'banner-row'

#     if n_clicks % 2 == 0:
#         style['display'] = ''
#         return style, 'banner-row'
#     else:
#         style['display'] = 'none'
#         return style, 'banner-row'


# @app.callback(
#     dash.dependencies.Output('secondary-menu-container', 'style'),
#     dash.dependencies.Input('tax-dropdown-button', 'n_clicks'),
#     prevent_initial_call=True
# )
# def toggle_secondary_menu(n):
#     # Initial hiding of the secondary menu
#     if n is None:
#         return {'display': 'none'}
#     # Toggle logic for even/odd clicks
#     if n % 2 == 0:
#         return {'display': 'none'}
#     else:
#         return {'display': 'block'}

# from dash.dependencies import Input, Output, State

# from dash.dependencies import Input, Output, State

# @app.callback(
#     [
#         Output('finvest-menu-container', 'style'),
#         Output('downward-triangle', 'style'),
#         Output('rightward-triangle', 'style')
#     ],
#     [
#         Input('finvest-dropdown-button', 'n_clicks'),
#         Input('url', 'pathname')
#     ],
#     [State('finvest-menu-container', 'style')],
#     prevent_initial_call=True
# )
# def toggle_secondary_menu2(n, pathname, current_style):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         # If not triggered by anything (initial load), hide the menu and show downward triangle
#         return {'display': 'none'}#, #{"width": "12px", "height": "10px", 'color':'white'}, {"width": "10px", "height": "12px", 'color':'white', 'display': 'none'}
#     else:
#         input_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
#         # If triggered by URL change, hide the menu and show downward triangle
#         if input_id == 'url':
#             return {'display': 'none'}, {"width": "12px", "height": "10px", 'color':'white'}, {"width": "10px", "height": "12px", 'color':'white', 'display': 'none'}
        
#         # If triggered by button click, toggle based on current state
#         if input_id == 'finvest-dropdown-button':
#             if current_style and current_style.get('display') == 'block':
#                 return {'display': 'none'}, {"width": "12px", "height": "10px", 'color':'white'}, {"width": "10px", "height": "12px", 'color':'white', 'display': 'none'}
#             else:
#                 return {'display': 'block'}, {"width": "12px", "height": "10px", 'color':'white', 'display': 'none'}, {"width": "10px", "height": "12px", 'color':'white'}

#     # Default return value
#     return {'display': 'none'}, {"width": "12px", "height": "10px", 'color':'white'}, {"width": "10px", "height": "12px", 'color':'white', 'display': 'none'}





# Define different styles for different screen sizes using CSS media queries

import socket
from contextlib import closing


def find_free_port():
     with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
       s.bind(('', 0))
       return s.getsockname()[1]

if __name__ == '__main__':
    app.run(debug=True, port=find_free_port())#, host='0.0.0.0') # host = 0.0.0.0 when running as a docker container)

