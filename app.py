from dash import Dash
import dash_bootstrap_components as dbc
from dash.long_callback import DiskcacheLongCallbackManager
import logging
import diskcache
from sys import version_info

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

# This is not supported in python 3.8
if version_info >= (3,9):
    logging.basicConfig(format='%(asctime)s %(message)s', filename='model.log', encoding='utf-8', level=logging.DEBUG)

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    long_callback_manager=long_callback_manager,
    )
app.config.suppress_callback_exceptions = True
server = app.server
