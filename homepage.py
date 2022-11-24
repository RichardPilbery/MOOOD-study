from dash import dcc, html 
import dash_bootstrap_components as dbc
from layouts import SHOW_BUTTON_STYLE, HIDE_BUTTON_STYLE


### Layout 1
home = html.Div([
    html.H1("Modelling 111 Demand", className='display-1'),
    html.Hr(),
    # create bootstrap grid 1Row x 2 cols
    dbc.Container([
        dbc.Row(
            [
                html.Div(
                    [
                    html.P('This site will enable to you run a discrete event simulation utilising a model of callers to the 111 telephone service who are triaged to a primary care disposition. Click on the Configuration link to get started.', className='lead'),
                    html.Img(src="assets/conceptual-model.png", width='100%')
                    ]
                ),                
            ],
        ), 
    ]),
])
