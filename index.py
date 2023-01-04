from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from app import app, server
from layouts import nav_bar, CONTENT_STYLE, footer 
from results import results
from homepage import home
from simulation import simulation
import callbacks

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Define basic structure of app:
# A horizontal navigation bar on the left side with page content on the right.
app.layout = html.Div([
    dcc.Location(id='url', refresh=False), #this locates this structure to the url
    dcc.Store(id='sim_start_date_store', storage_type='local'),
    dcc.Store(id='sim_duration_store', storage_type='local'),
    dcc.Store(id='warm_up_duration_store', storage_type='local'),
    dcc.Store(id='store-num-runs', storage_type='local'),
    dcc.Store(id='what_if_sim_run_store', storage_type='local'),
    dcc.Store(id='transition_type_store', storage_type='local'),
    dcc.Store(id='ia_time_store', storage_type='local'),
    html.Div([
        nav_bar(),
        html.Div(id='page-content',style=CONTENT_STYLE), #we'll use a callback to change the layout of this section 
    ]),
    html.Div([
        footer()
    ], className='w-100 container-fluid bg-secondary mt-5', style={"zIndex":'1', 'position':'relative', 'minHeight':'200px'})
])

# This callback changes the layout of the page based on the URL
# For each layout read the current URL page "http://127.0.0.1:8080/pagename" and return the layout
@app.callback(Output('page-content', 'children'), #this changes the content
              [Input('url', 'pathname')]) #this listens for the url in use
def display_page(pathname):
    if pathname == '/':
        return home
    elif pathname == '/home':
        return home
    elif pathname == '/simulation':
        return simulation
    elif pathname == '/results':
         return results
    else:
        return '404' #If page not found return 404


if __name__ == "__main__":
    app.run(debug=True)
