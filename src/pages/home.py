import dash
from dash import html, dcc

dash.register_page(__name__, path="/")

card_style = {
    "backgroundColor": "#F9F9F9",
    "borderRadius": "8px",
    "padding": "16px",
    "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)",
}

title_link_style = {
    "display": "inline-block",
    "fontSize": "2rem",
    "textAlign": "center",
}

wide_title_link_style = {
    "display": "inline-block",
    "fontSize": "2rem",
    "width": "75%",
    "textAlign": "center",
}

layout = html.Div(
    [
        #html.Div(className="beforediv"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.A(
                                            html.Span("US Economy"),
                                            href="/economy",
                                            className="headers",
                                            style=title_link_style,
                                        )
                                    ],
                                    style={"textAlign": "center"},
                                ),
                                dcc.Markdown("Key data representing the state of the US economy"),
                            ],
                            className="page-intro",
                            style=card_style,
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.A(
                                            [html.Span("Bond Market")],
                                            href="/yield_curves",
                                            className="headers",
                                            style=wide_title_link_style,
                                        )
                                    ],
                                    style={"textAlign": "center"},
                                ),
                                dcc.Markdown("US and Norwegian Government Bond data"),
                            ],
                            className="page-intro",
                            style=card_style,
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.A(
                                            [html.Span("Algorithm")],
                                            href="/portfolio-daily",
                                            className="headers",
                                            style=wide_title_link_style,
                                        )
                                    ],
                                    style={"textAlign": "center"},
                                ),
                                dcc.Markdown("Overview of optimized fundamental stock selection algorithm"),
                            ],
                            className="page-intro",
                            style=card_style,
                        ),
                    ],
                    className="page-intros fadeinelement home-intro-grid",
                ),
                html.Br(),
                
                html.Br(),
            ],
            className="page-intros home-hero-section",
            style={
                "backgroundImage": 'url("/assets/pretty3.JPG")',
                "backgroundSize": "cover",
                "backgroundPosition": "center",
                "backgroundRepeat": "no-repeat",
                "minHeight": "100vh",
                "position": "relative",
            },
        ),
    ]
)