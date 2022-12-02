from turtle import width
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

# Mostly stolen from here: https://towardsdatascience.com/callbacks-layouts-bootstrap-how-to-create-dashboards-in-plotly-dash-1d233ff63e30

NAVBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "backgroundColor": "#f8f9fa",
}

CONTENT_STYLE = {
    "top":0,
    "marginTop":'2rem',
    "marginLeft": "18rem",
    "marginRight": "2rem",
}

SHOW_BUTTON_STYLE = {
    "display": "block"
}

HIDE_BUTTON_STYLE = {
    "display": "none"
}

DROPDOWN_STYLE = {
    "marginBottom":"20px"
}

NETWORK_GRAPH_STYLE = {
    "border": "2px solid black",
    'width':'90%', 
    'height':'600px'
}


#####################################
# Create Auxiliary Components Here
#####################################

def nav_bar():
    """
    Creates Navigation bar
    """
    navbar = html.Div(
    [
        html.Div(html.Img(src="assets/telephone-call.png", width="50%", className="img-fluid"),className="text-center"),
        
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/home",active="exact", external_link=True),
                dbc.NavLink("Configuration", href="/simulation",active="exact", external_link=True),
                dbc.NavLink("Results", href="/results", active="exact", external_link=True)
            ],
            pills=True,
            vertical=True
        ),
    ],
    style=NAVBAR_STYLE,
    )  
    return navbar

def footer():
    footer = dbc.Container(
        [   
            dbc.Row([
                dbc.Col([
                    html.P(' ')
                ]),
                dbc.Col([
                    html.P(' ')
                ]),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([html.A(href='http://www.twitter.com/999CPD', className='fa-brands fa-twitter display-3 text-light', style={'textDecoration':'none'})]),
                        dbc.Col([html.A(href='https://github.com/RichardPilbery/MOOOD-study', className='fa-brands fa-github display-3 text-light', style={'textDecoration':'none'})]),
                        dbc.Col([html.A(href='https://uk.linkedin.com/in/richard-pilbery-4138286b', className='fa-brands fa-linkedin display-3 text-light', style={'textDecoration':'none'})])
                    ]),
                    dbc.Row([
                        html.Div([
                            html.Span('Icons made by '),
                            html.A('Freepik', href="https://www.flaticon.com/authors/freepik", className='text-light'),
                            html.Span(' from '),
                            html.A('www.flaticon.com', href="https://www.flaticon.com/", className='text-light')
                        ], className="small pt-5 text-light")

                    ])

                ])
            ], className='pt-4')
        ], className='pt-4'
    )

    return footer


##############################
# Large summary box with data

def summary_dash(header = 'GP', id = None):
    text_colour = "text-success"
    if header in ['ED', '999']:
        text_colour = "text-warning"
    elif header == 'None':
        text_colour = 'text-danger'
    elif header == '111':
        text_colour = 'text-primary'

    #print(id)
    wi_result = html.Div([
                html.Span('What if analysis: '),
                html.Span(id=f"wi-{id}"),
                html.Span('%')
            ], style={'display':'block'}, id=f"wi-visible-{id}")

    return html.Div(
        [
            html.H2(header, className=f"display-5 {text_colour}"),
            html.Hr(className="my-2"),
            html.Span(
                className="display-5 text_dark", id=id
            ),
            html.Span('%',
                className="display-5 text_dark"
            ),
            wi_result,
            html.Div([
                html.Span('2021 Connected Yorkshire: '),
                html.Span(id=f"cb-{id}"),
                html.Span('%')
            ])
        ],
        className="h-100 p-3 bg-light border rounded-3 text-center",
    )



def tab_outcome_content():
    return html.Div([
            html.H4('First patient contact after index 111 call', className='pt-4 display-5'),
            dbc.Row(
                [
                    dbc.Col(
                       summary_dash('GP', 'gp-summary-stat')
                    ),
                    dbc.Col(
                        summary_dash('ED', 'ed-summary-stat')
                    ),
                    dbc.Col(
                       summary_dash('999', '999-summary-stat')
                    ),
                    dbc.Col(
                        summary_dash('111', '111-summary-stat')
                    ),
                    dbc.Col(
                        summary_dash('None', 'none-summary-stat')
                    ),
                ],
                className="pb-4",
            ),
            html.H4('Quarterly counts stratified by service', className='display-4'),
            dcc.Graph(id='quarterly_counts_figure', className='pb-4'),
            html.Div(id='quarterly_counts_table', className='pb-4'),
            dbc.Row([
                dbc.Row([
                    dbc.Col(
                        html.H4('Network graph', className='display-5')
                    ),
                    dbc.Col([       
                        html.Button('Download Network graph data', id="btn-download-network-analysis", className="btn btn-primary mt-3"),
                        dcc.Download(id="download-network-analysis")
                    ])
                ]),
                cyto.Cytoscape(
                    id='network-graph',
                    style=NETWORK_GRAPH_STYLE,
                    elements={},
                    layout={'name': 'cose',
                            'fit' : True,
                            'nodeRepulsion': '9000',
                            'gravityRange': '10.0',
                            'nestingFactor': '0.8',
                            'edgeElasticity': '150',
                            'idealEdgeLength': '500',
                            'nodeDimensionsIncludeLabels': True,
                            'numIter': '6000',
                            },
                    minZoom=0.5,
                    maxZoom=2,
                    responsive=True,
                    stylesheet=[
                        {'selector': 'node',
                            'style': {
                                    'width': '20px',
                                    'height': '20px',
                                    'background-color': '#1267D6',
                                    'content': 'data(label)',
                                    'font-size': '40px',
                                    'text-opacity': '0.5'
                                }
                            },
                        {'selector': 'edge',
                            'style': {
                                'curve-style': 'unbundled-bezier',
                                'label' : 'data(weight)',
                                'text-margin-y': 0,
                                'color' : 'black',
                                'font-size' : '30px',
                                'text-outline-color': 'white',
                                'text-outline-width': '5px',
                                'text-rotation': 'autorotate',
                                'width' : 'data(edgeline)',
                                'line-color': 'gray',
                                'target-arrow-color': '#444',
                                'target-arrow-shape': 'triangle',
                                'arrow-scale' : 1.5,
                            }
                        }
                    ]
                )
            ]),
            dbc.Row([
            html.H4('Sankey diagram', className='display-5'),
            html.Div([
                html.H4('Simulated data',className='display-6'),
                dcc.Graph(id='sankey-diagram')
            ]),
            html.Div([
                html.H4('Simulated data - Timely GP response',className='display-6'),
                dcc.Graph(id='wi-sankey-diagram')
            ]),
            html.Div([
                html.H4('Connected Bradford 2021 Sankey diagram',className='display-6'),
                dcc.Graph(id='cb-sankey-diagram')
            ]),
            html.Hr(),
        ], class_name='pt-4')
        ])



def tab_calldata_content():
    return html.Div([
        html.H4('Connected Yorkshire 2021 call data', className='display-4'),
        dbc.Row([
            html.H4('111 call volume by hour and day of week', className='display-6'),
            html.Div([
                dcc.Graph(id='ooo-call-volume')
            ]),
            html.Hr(),
        ]),
        dbc.Row([
            html.H4('Simulated 111 call volume by hour and day of week', className='display-6'),
            html.Div([
                dcc.Graph(id='sim-ooo-call-volume')
            ]),
            html.Hr(),
        ])
    ])


def tab_ed_content():
    return html.Div([
        html.H4('Emergency Department Activity', className='display-4'),
        dbc.Row([
            html.H4('ED attendance volume by hour and day of week', className='display-6'),
            html.Div([
                dcc.Graph(id='ed-attend-volume')
            ]),
            html.Hr(),
        ]),
        dbc.Row([
            html.H4('Simulated ED attendance volume by hour and day of week', className='display-6'),
            html.Div([
                dcc.Graph(id='sim-ed-attend-volume')
            ]),
            html.Hr(),
        ]),
        dbc.Row([
            html.H4('Proportion of ED attendances classified as avoidable', className='display-6'),
            html.Div(id='prop_avoidable_admissions_table', className='pb-4'),
            html.H4('ED attendances stratified by avoidable admission', className='display-6'),
            html.Div(id='avoidable_admissions_table', className='pb-4'),
        ])

    ])