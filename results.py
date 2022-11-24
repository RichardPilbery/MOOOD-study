
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from layouts import DROPDOWN_STYLE, tab_outcome_content, tab_calldata_content, tab_ed_content
from app import app

### Layout 2
results = html.Div(
    [
        html.H2('Simulation Results', className='display-2'),
        html.Hr(),
        dbc.Container(
            [
                dbc.Row(
                    [
                        html.Div([
                            dcc.Dropdown(id='dropdown-run-number')
                        ], style=DROPDOWN_STYLE),
                        html.H4([
                            html.Span('Total number of index 111 calls with a primary care triage outcome: '),
                            html.Span(id='total-summary-stat'),
                        ], className='lead'),
                        html.P([
                            html.Span('2021 Connected Yorkshire total: '),
                            html.Span(id='cb-total-summary-stat')
                        ]),
                        html.Hr(),
                    ],
                ),
                dcc.Tabs(id="tabs-results", value='tab-outcome', children=[
                    dcc.Tab(label='Outcome', value='tab-outcome'),
                    dcc.Tab(label='Call data', value='tab-calldata'),
                    dcc.Tab(label='ED activity', value='tab-ed'),
                ]),
                html.Div(id='tabs-results-content')
            ]
        )
    ]
)

@app.callback(
    Output('tabs-results-content', 'children'),
    Input('tabs-results', 'value')
)
def render_tab(tab):
    if tab == 'tab-outcome':
        return tab_outcome_content()
    elif tab == 'tab-calldata':
        return tab_calldata_content()
    elif tab == 'tab-ed':
        return tab_ed_content()