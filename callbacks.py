import json
from dash import html, Input, Output, State, dcc, dash_table
from app import app, long_callback_manager
from layouts import SHOW_BUTTON_STYLE, HIDE_BUTTON_STYLE
import logging
from oneoneonedes import parallelProcess, prepStartingVars
import pandas as pd
import plotly.express as px
import numpy as np
from utilities import create_cyto_graph, summary_stats, using_nth, quarterly_counts, draw_Sankey, draw_CB_Sankey, prep_draw_Sankey, vol_fig, prep_admissions_table
from g import G
from os.path import exists
from dash.exceptions import PreventUpdate

buttonClickCount = 0

# 3. When trigger sim is set to value = 1, then simulation will run
# When it has finished, sim_complete is set to 1
@app.long_callback(
    output = (
        Output('sim_complete', 'value')
    ),
    inputs = (
        Input('run_sim', 'n_clicks'),
        State('sim_duration', 'value'),
        State('warm_up_time', 'value'),
        State('number_of_runs', 'value'),
        State('sim_start_date', 'date'),
        State('what_if_sim_run_value', 'value'),
        State('transition_type_value', 'value'),
        State('ia_time_value', 'value')
    ),
    running = [
        (Output('submit_button', 'style'), HIDE_BUTTON_STYLE, SHOW_BUTTON_STYLE),
        (Output('sim_run_button', 'style'), SHOW_BUTTON_STYLE, HIDE_BUTTON_STYLE),
    ],
    manager = long_callback_manager,
    prevent_initial_call = True
)
def configSim(run_sim, sim_duration, warm_up_time, number_of_runs, sim_start_date, what_if_sim_run, transition_type_value, ia_time_value):
    # Run the sim
    logging.debug("Run the sim")

    global buttonClickCount

    if run_sim > buttonClickCount:
        logging.debug('run_sim > buttonClickCount')
        print('run_sim > buttonClickCount')
        print(f"Sim duration {sim_duration} and warm up {warm_up_time}")
        buttonClickCount = run_sim
        prepStartingVars(
            number_of_runs = number_of_runs, 
            warm_up_time = warm_up_time * 1440, 
            sim_duration = sim_duration * 1440,
            sim_start_date = f"{sim_start_date} 00:00:00",
            what_if_sim_run = what_if_sim_run,
            transition_type = transition_type_value,
            ia_type = ia_time_value
        )
        parallelProcess()

        return 1
    else:
        return 0

# Save what if radio button into storage
@app.callback(
    Output('sim_start_date_store', 'data'),
    Output('sim_duration_store', 'data'),
    Output('warm_up_duration_store', 'data'),
    Output('store-num-runs', 'data'),
    Output('what_if_sim_run_store', 'data'),
    Output('transition_type_store', 'data'),
    Output('ia_time_store', 'data'),

    Input('run_sim', 'n_clicks'),

    State('sim_start_date', 'date'),
    State('sim_duration', 'value'),
    State('warm_up_time', 'value'),
    State('number_of_runs', 'value'),
    State('what_if_sim_run_value', 'value'),
    State('transition_type_value', 'value'),
    State('ia_time_value', 'value')

)
def save_to_local_storage(run_sim, sim_start_date, sim_duration, warm_up_time, number_of_runs, what_if_sim_run_value, transition_type_value, ia_time_value):
    # Sim complete is set to -1 on a refresh
    # We only want to update local storage if the
    # sim button has been pressed and so value has been set to 0
    print(f"Submit button clicked, updating local storage {run_sim}")
    global buttonClickCount

    if run_sim > buttonClickCount:
        return [sim_start_date, sim_duration, warm_up_time, number_of_runs, what_if_sim_run_value, transition_type_value, ia_time_value]
    else:
        raise PreventUpdate




# Update Summary stats on Results > Outcome page
@app.callback(
    Output('total-summary-stat', 'children'), 
    Output('gp-summary-stat', 'children'), 
    Output('ed-summary-stat', 'children'), 
    Output('999-summary-stat', 'children'), 
    Output('111-summary-stat', 'children'), 
    Output('none-summary-stat', 'children'), 
    Output('wi-gp-summary-stat', 'children'), 
    Output('wi-ed-summary-stat', 'children'), 
    Output('wi-999-summary-stat', 'children'), 
    Output('wi-111-summary-stat', 'children'), 
    Output('wi-none-summary-stat', 'children'), 
    Output('cb-total-summary-stat', 'children'), 
    Output('cb-gp-summary-stat', 'children'),
    Output('cb-ed-summary-stat', 'children'), 
    Output('cb-999-summary-stat', 'children'), 
    Output('cb-111-summary-stat', 'children'), 
    Output('cb-none-summary-stat', 'children'), 
    Output('wi-visible-gp-summary-stat', 'style'), 
    Output('wi-visible-ed-summary-stat', 'style'), 
    Output('wi-visible-999-summary-stat', 'style'), 
    Output('wi-visible-111-summary-stat', 'style'), 
    Output('wi-visible-none-summary-stat', 'style'), 

    Output('sim_start_date_state', 'children'),
    Output('sim_duration_state', 'children'),
    Output('warm_up_duration_state', 'children'),
    Output('store-num-runs_value', 'children'),
    Output('transition_type_state', 'children'),
    Output('ia_time_state', 'children'),

    Input('dropdown-run-number', 'value'),

    State('sim_start_date_store', 'data'),
    State('sim_duration_store', 'data'),
    State('warm_up_duration_store', 'data'),
    State('store-num-runs', 'data'),
    State('what_if_sim_run_store', 'data'),
    State('transition_type_store', 'data'),
    State('ia_time_store', 'data'),
    
)
def summary_stat_page_content(run_number, sim_start_date, sim_duration, warm_up_time, number_of_runs, what_if_sim_run, transition_type, ia_time):
    values_dict = summary_stats(what_if_sim_run, run_number)
    # print(values_dict.keys())

    wi_visibility = {'display':'block'}
    
    if what_if_sim_run == 'No':
        wi_visibility = {'display':'none'} 
        values_dict['wi_GP'] = None
        values_dict['wi_ED'] = None
        values_dict['wi_999'] = None
        values_dict['wi_IUC'] = None
        values_dict['wi_End'] = None

    return values_dict['total'], values_dict['GP'], values_dict['ED'], values_dict['999'], values_dict['IUC'], values_dict['End'], values_dict['wi_GP'], values_dict['wi_ED'], values_dict['wi_999'], values_dict['wi_IUC'], values_dict['wi_End'],values_dict['cb_total'], values_dict['cb_GP'], values_dict['cb_ED'], values_dict['cb_999'], values_dict['cb_IUC'], values_dict['cb_End'], wi_visibility, wi_visibility, wi_visibility, wi_visibility, wi_visibility, sim_start_date, sim_duration, warm_up_time, number_of_runs, transition_type, ia_time




# Update figure based on run number chosen
@app.callback(
    Output('age-dist', 'figure'), 
    State('what_if_sim_run_store', 'data'),
    Input('dropdown-run-number', 'value'),
)
def fig_age_dist(what_if_sim_run, run_number = 999):
    # TODO: Handle case when there is no CSV file yet
    df = pd.read_csv(G.all_results_location) 
    
    if (what_if_sim_run == 'Yes' and exists(G.wi_all_results_location)):
        df = pd.read_csv(G.wi_all_results_location)
    
    print('Loaded figures python')

    if run_number is None:
        run_number = 999

    print(f"fig_age_dist called and run number is {run_number}")

    thetitle = "Age distribution for mean of all runs"

    age_wrangle = (
        df[['run_number','P_ID', 'age', 'sex', 'GP']]
        .drop_duplicates()
        .groupby(['run_number','age'], as_index=False)['P_ID']
        .count()
    )

    if(run_number < 999):
        # Just return a single run's data

        age_wrangle = (
            df[df['run_number'] == run_number]
            .filter(['P_ID','age'])
            .drop_duplicates()
            .groupby(['age'], as_index=False)['P_ID']
            .count()
        )
        thetitle = f"Age distribution for run number {run_number + 1}"

    fig = px.histogram(
        age_wrangle, 
        x='age', 
        y='P_ID',
        histfunc='avg',
        nbins=100,
        labels={
            'age': 'Patient age in years',
            'P_ID' : 'Number of patients'
        },
        template='plotly_white',
        title = thetitle
    )
    fig.layout['yaxis_title'] =  "Number of patients"
    fig.update_traces(marker_line_width=0.5,marker_line_color="#333")
    return fig

# Populate age dropdown based on the number of runs available
@app.callback(
    Output('dropdown-run-number', 'options'), 
    Output('dropdown-run-number', 'value'),
    Input('url', 'pathname'),
    State('store-num-runs', 'data'),
)
def dropdown_run_number(url, num_runs):
    print('Dropdown run number')

    if num_runs is None:
        num_runs = 1
    else:
        num_runs = int(num_runs)

    keys = list(range(0,num_runs))
    thelist = [{'label': 'Average (mean) of all runs', 'value': 999}]
    for i in keys:
        thelist.append({'label':i+1, 'value':i})

    return thelist, 999


# Update figure to show call activity
@app.callback(
    Output('ooo-call-volume', 'figure'), 
    State('what_if_sim_run_store', 'data'),
    Input('dropdown-run-number', 'value')
)
def fig_call_vol(what_if_sim_run, run_number = 999):
    # TODO: Handle case when there is no CSV file yet
    df = pd.read_csv(G.cb_111_call_volumes)
    thetitle = f"111 call volumes by hour and day of week"

    if run_number is None:
        run_number = 999

    if run_number == 999:
        df1 = df.groupby(['day','hour'], as_index=True)['P_ID'].agg('mean').reset_index()
    else:
        df1 = df

    fig = vol_fig(df1, thetitle, run_number, sim = False)

    return fig

# Update figure to show simulated call activity
@app.callback(
    Output('sim-ooo-call-volume', 'figure'), 
    State('what_if_sim_run_store', 'data'),
    Input('dropdown-run-number', 'value')
)
def sim_fig_call_vol(what_if_sim_run, run_number = 999):
    # TODO: Handle case when there is no CSV file yet
    df = pd.read_csv(G.all_results_location)
    thetitle = f"Simulated 111 call volumes for run number {run_number + 1} by hour and day of week"

    if run_number is None:
        run_number = 999
        thetitle = f"Mean simulated 111 call volumes by hour and day of week"

    fig = vol_fig(df[df['activity'] == 'Index_IUC'], thetitle, run_number)

    return fig




# Update ED figure to show call activity
@app.callback(
    Output('ed-attend-volume', 'figure'), 
    State('what_if_sim_run_store', 'data'),
    Input('dropdown-run-number', 'value')
)
def ed_attend_vol(what_if_sim_run, run_number = 999):
    # TODO: Handle case when there is no CSV file yet
    df = pd.read_csv(G.cb_ed_attedance_volumes)
    thetitle = f"ED attendance volume by hour and day of week"

    if run_number is None:
        run_number = 999

    fig = vol_fig(df, thetitle, run_number, sim = False)

    return fig

# Update figure to show ED simulated call activity
@app.callback(
    Output('sim-ed-attend-volume', 'figure'), 
    State('what_if_sim_run_store', 'data'),
    Input('dropdown-run-number', 'value')
)
def sim_ed_attend_vol(what_if_sim_run, run_number = 999):
    # TODO: Handle case when there is no CSV file yet
    df = pd.read_csv(G.all_results_location, low_memory=False)
    df1 = df[(df.status == 'start') & (df.activity == 'ED')]
    thetitle = f"Simulated ED attendance volume for run number {run_number + 1} by hour and day of week"

    if run_number is None:
        run_number = 999
        thetitle = f"Mean ED attendance volume by hour and day of week"

    fig = vol_fig(df1, thetitle, run_number)

    return fig


# Show network graph
@app.callback(
    Output('network-graph', 'elements'), 
    Input('dropdown-run-number', 'value'),
    State('what_if_sim_run_store', 'data')
)
def network_graph_draw(run_number, what_if_sim_run):
    elements = create_cyto_graph(what_if_sim_run, run_number)

    return elements

    
# Show Sankey diagram
@app.callback(
    Output('sankey-diagram', 'figure'), 
    Output('wi-sankey-diagram', 'figure'), 
    Output('cb-sankey-diagram', 'figure'), 
    State('what_if_sim_run_store', 'data'),
    Input('dropdown-run-number', 'value'),
)
def sankey(what_if_sim_run, run_number = 999):
    # TODO: Handle case when there is no CSV file yet
    df = pd.read_csv('data/all_results.csv', low_memory=False)

    # TODO: Probably need to do something more graceful than just set to df
    wi_df = pd.read_csv(G.wi_all_results_location) if exists(G.wi_all_results_location) else df

    if run_number < 999:
        df = df[df['run_number'] == run_number]

    sorted_labels, colorList, short_name, source_numbers, target_numbers, counts2 = prep_draw_Sankey(df, run_number)
    wi_sorted_labels, wi_colorList, wi_short_name, wi_source_numbers, wi_target_numbers, wi_counts2 = prep_draw_Sankey(wi_df, run_number)

    return [
        draw_Sankey(sorted_labels, colorList, short_name, source_numbers, target_numbers, counts2),
        draw_Sankey(wi_sorted_labels, wi_colorList, wi_short_name, wi_source_numbers, wi_target_numbers, wi_counts2 ),
        draw_CB_Sankey()
    ]

# Download callbacks

@app.callback(
    Output("download-network-analysis", "data"),
    Input("btn-download-network-analysis", "n_clicks"),
    State('what_if_sim_run_store', 'data'),
    prevent_initial_call=True,
)
def downloadNetworkAnalysis(n_clicks, what_if_sim_run):
    return dcc.send_file("./data/network_graph.csv") if (what_if_sim_run == 'No') else dcc.send_file("./data/wi_network_graph.csv")



# Quarterly counts stratified by service


# Update figure to show call activity
@app.callback(
    Output('quarterly_counts_figure', 'figure'), 
    State('what_if_sim_run_store', 'data'),
    Input('dropdown-run-number', 'value')
)
def fig_qtr_counts(what_if_sim_run, run_number = 999):
    # TODO: Handle case when there is no CSV file yet
    cb_df = pd.read_csv(G.cb_quarterly_counts)
    df = pd.read_csv(G.all_results_location, low_memory=False)

    if (what_if_sim_run == 'Yes' and exists(G.wi_all_results_location)):
        wi_df = pd.read_csv(G.wi_all_results_location)
        wi_df1 = wi_df if run_number == 999 else wi_df[wi_df['run_number'] == run_number]
        wi_df2 = quarterly_counts(wi_df1, 'What if')

    else:
        wi_df2 = None

    df1 = df if run_number == 999 else df[df['run_number'] == run_number]
    df2 = quarterly_counts(df1)

    figdata = pd.concat([cb_df, df2, wi_df2], axis=0)
    #print(figdata)
    figdata2 = figdata.sort_values(by=['data','quarter', 'count'], ascending=True)

    fig = px.bar(figdata2, x='activity', y='count', color='data', color_discrete_sequence=px.colors.qualitative.D3, facet_col='quarter', facet_col_wrap=2, text_auto=True, barmode='group', height=800, log_y=True)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    return fig

# Tabulate quarterly count data
@app.callback(
    Output('quarterly_counts_table', 'children'), 
    State('what_if_sim_run_store', 'data'),
    Input('dropdown-run-number', 'value')
)
def fig_qtr_tab_counts(what_if_sim_run, run_number = 999):
    print('Inside table callback')
    # TODO: Handle case when there is no CSV file yet
    cb_df = pd.read_csv(G.cb_quarterly_counts)
    df = pd.read_csv(G.all_results_location, low_memory=False)

    if (what_if_sim_run == 'Yes' and exists(G.wi_all_results_location)):
        wi_df = pd.read_csv(G.wi_all_results_location)
        wi_df1 = wi_df if run_number == 999 else df[df['run_number'] == run_number]
        wi_df2 = quarterly_counts(wi_df1, 'What if')

    else:
        wi_df2 = None

    df1 = df if run_number == 999 else df[df['run_number'] == run_number]
    df2 = quarterly_counts(df1)

   # print(f"wi_id is None: {wi_df2 is None}")
    tabdata = pd.concat([cb_df, df2], axis=0, ignore_index=True) if wi_df2 is None else pd.concat([cb_df, df2, wi_df2], axis=0, ignore_index=True) 
    #tabdata2 = tabdata.sort_values(by=['data','quarter', 'count'], ascending=True)
    #print(tabdata)
    #tabdata['combo'] = tabdata['count'] if (tabdata['data'] == 'cYorkshire2021') else tabdata['mean'].astype(str) + ' (' + tabdata['Lower_95_CI'].astype(str) + '--' + tabdata['Lower_95_CI'].astype(str) + ')'
    tabdata['combo'] = np.where(tabdata['data'] == 'cYorkshire2021', tabdata['count'], tabdata['mean'].astype(str) + ' (' + tabdata['Lower_95_CI'].astype(str) + '-' + tabdata['Upper_95_CI'].astype(str) + ')')
    tabdata2 = tabdata.pivot(index=["activity","quarter"], columns="data", values="combo").reset_index()

    #print(tabdata2)

    table_cols = tabdata2.columns
    #print(table_cols)
    table = tabdata2.loc[:, table_cols].to_dict('rows')

    columns = [{"name": i, "id": i,} for i in (table_cols)]

    #print(dash_table.DataTable(data=table, columns=columns))

    return dash_table.DataTable(
        data=table, 
        columns=columns, 
        export_format="csv",
        style_table={
            'color': 'black',
            'backgroundColor': 'white',
            'font-family': 'Arial, Helvetica, sans-serif'
        },
        style_data={
            'font-family': 'Arial, Helvetica, sans-serif'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(220, 220, 220)',
            }
        ],
        style_header={
            'backgroundColor': 'rgb(210, 210, 210)',
            'fontWeight': 'bold',
            'font-family': 'Arial, Helvetica, sans-serif'
        }
    )



# Tabulate Avoidable admission data
@app.callback(
    Output('avoidable_admissions_table', 'children'), 
    State('what_if_sim_run_store', 'data'),
    Input('dropdown-run-number', 'value')
)
def avoid_adm_tab_counts(what_if_sim_run, run_number = 999):
    
    table_prep = prep_admissions_table(what_if_sim_run, run_number, type='count')

    return dash_table.DataTable(data=table_prep[0], columns=table_prep[1], export_format="csv")

# Tabulate Avoidable admission data
@app.callback(
    Output('prop_avoidable_admissions_table', 'children'), 
    State('what_if_sim_run_store', 'data'),
    Input('dropdown-run-number', 'value')
)
def avoid_adm_tab_counts(what_if_sim_run, run_number = 999):
    
    table_prep = prep_admissions_table(what_if_sim_run, run_number, type='prop')

    return dash_table.DataTable(data=table_prep[0], columns=table_prep[1], export_format="csv")
