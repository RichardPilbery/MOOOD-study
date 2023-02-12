from datetime import date
from dash import dcc, html 
import dash_bootstrap_components as dbc
from layouts import SHOW_BUTTON_STYLE, HIDE_BUTTON_STYLE
from g import G


### Layout 1
simulation = html.Div([
    html.H2("Configure Simulation", className='display-2'),
    html.Hr(),
    # create bootstrap grid 1Row x 2 cols
    dbc.Container([
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                            html.H4('Instructions', className='display-4'),
                            #create tabs
                            html.P("This site will enable you run a discrete event simulation, to model demand arising from 111 calls triaged with a primary care dispostion. Adjust the settings to configure the simulation.", className='lead'),
                            html.P("Once the simulation has completed, visit the Results page to view the model outputs.", className='lead')
                            ]
                        ),
                    ],
                    width=6 #half page
                ),
                
                dbc.Col(
                    [
                        html.Div([
                            html.P('Simulation start date', id='sim_start_date_label', className='lead'),
                            dcc.DatePickerSingle(
                                id='sim_start_date',
                                date=G.start_dt,
                                display_format="DD-MM-YYYY",
                                first_day_of_week=1
                            ),
                        ], className="pb-4"),

                        html.P('Sim duration in days', id='sim_duration_label', className='lead'),
                        dcc.Slider(
                            7,
                            365,
                            step=1,
                            value=7,
                            marks=None,
                            id='sim_duration',
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),

                        html.P('Warm up duration (days)', id='warm_up_duration_label', className='lead'),
                        dcc.Slider(
                            0,
                            2,
                            step=1,
                            value=1,
                            marks=None,
                            id='warm_up_time',
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),

                        html.P('Number of runs', id='number_of_runs_label', className='lead'),
                        dcc.Slider(
                            1,
                            100,
                            step=1,
                            value=3,
                            marks=None,
                            id='number_of_runs',
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Div([
                            html.P('Run \'What if?\' simulation', id="what_if_sim_label", className='lead'),
                            html.P('This option will adjust the simulation so that all callers will have contact with a GP', className='small'),
                            dcc.RadioItems(['No', 'Yes'], 'Yes', id="what_if_sim_run_value", inline=False, inputStyle={"marginRight": "5px"}, labelStyle={"marginRight": "10px"}, className='lead'),
                        ], className="pt-5"),


                        html.Div([
                            html.P('Transition type', id="transition_type_label", className='lead'),
                            html.P('This option enables you to select the type of transition (how patient movement from one service to the next) will be calculated', className='small'),
                            dcc.RadioItems([
                                {'label' : 'Base', 'value' : 'base'}, 
                                {'label' : 'Base + Yearly Quarters', 'value': 'qtr'},
                                {'label' : 'ML', 'value' : 'tree'}
                            ], 'qtr', id="transition_type_value", inline=False, inputStyle={"marginRight": "5px"}, labelStyle={"marginRight": "10px"}, className='lead'),
                        ], className="pt-5"),
                        html.Div([
                            html.P('Inter-arrival time', id="ia_time_label", className='lead'),
                            html.P('This option determines the method used to calculate call arrival times. Base distributions are sampled from exponential distribution. Mean uses the mean interarrival time with no sampling.', className='small'),
                            dcc.RadioItems([
                                {'label' : 'Base', 'value' : 'base'},
                                {'label' : 'Base + max rates', 'value' : 'max'},
                                {'label' : 'Base + min/max rates', 'value' : 'minmax'},
                                {'label' : 'Mean', 'value' : 'mean'}
                            ], 'base', id="ia_time_value", inline=False, inputStyle={"marginRight": "5px"}, labelStyle={"marginRight": "10px"}, className='lead'),
                        ], className="py-5"),
                        html.Div(
                            id="submit_button",
                            children=[
                                dbc.Button(
                                    "Run Simulation", id="run_sim", class_name="me-2 btn-lg", outline=True, color="primary",
                                    n_clicks = 0,
                                ),
                            ],
                            style = SHOW_BUTTON_STYLE
                        ),
                        html.Div(
                            id="sim_run_button",
                            children=[
                                dbc.Button(
                                    [dbc.Spinner(size="sm"), " Simulation running..."],
                                    id="sim_running",
                                    color="primary",
                                    disabled=True,
                                ),
                            ],
                            style = HIDE_BUTTON_STYLE
                        ),

                        html.Div(
                            id="sim_block",
                            children=[
                                dcc.Input(
                                    id="sim_running", 
                                    type="hidden", 
                                    value=0,
                                ),
                                dcc.Input(
                                    id="sim_complete", 
                                    type="hidden", 
                                    value=-1,
                                ),
                                dcc.Input(
                                    id="toggle_button", 
                                    type="hidden", 
                                    value=0,
                                )
                            ],
                            style = {'display':'none'}
                        ),


                    ],
                    width=6 #half page
                )
                
            ],
        ), 
    ]),
])
