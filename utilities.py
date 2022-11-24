import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# This function will create a json array for cytoscape
# The nodes and edges will be extracted from the csv files and the elements returned.
def create_cyto_graph(what_if_sim_run, run_number = 999):
    nodeData = pd.read_csv("data/cb_node_list.csv", low_memory=False)
    #edges_1 = pd.read_csv("cb_edge_list.csv", low_memory=False)
    edgeData = pd.read_csv("data/network_graph.csv", low_memory=False, index_col=False) if (what_if_sim_run == 'No') else pd.read_csv("data/wi_network_graph.csv", low_memory=False, index_col=False)

    if run_number == 999:
        edgeData2 = edgeData.groupby(['source', 'target'], as_index=False).agg(weight = pd.NamedAgg(column='weight', aggfunc='mean'))
        # Add run_number column back in to avoid errors
        edgeData2.insert(2, 'run_number', 9999999)
        edgeData2['weight'] = edgeData2['weight'].round(0)
        edgeData2 = edgeData2[edgeData2['weight'] > 0]
        #print(edgeData2.head())
    else:
        edgeData2 = edgeData[edgeData['run_number'] == run_number].copy()
        #print(edgeData2.head())
        edgeData2['weight'] = edgeData2['weight'].round(0)
        #print(edgeData2.head())
        edgeData2 = edgeData2[edgeData2['weight'] > 0]
        #print(edgeData2.head())

    nodes_list = list()
    for i in range(len(nodeData)):
        node = {
                "data": {"id": nodeData.iloc[i,0].astype(str), 
                         "label": nodeData.iloc[i,1],
                },
            }
        nodes_list.append(node)
    
    edges_list = list()
    for j in range(len(edgeData2)):
        edge = {
                "data": 
                    {
                        "source": edgeData2.iloc[j,0].astype(str), 
                         "target": edgeData2.iloc[j,1].astype(str),
                         "weight": edgeData2.iloc[j,3],
                         #"edgeline" : edges_1.iloc[j,4]/1000 if edges_1.iloc[j,4] > 14000 else edges_1.iloc[j,4]/500,
                         "edgeline" : 2,
                         # "weightline" : 0.5 if edges_1.iloc[j,4] > 14000 else 1}
                    }
            }
        edges_list.append(edge)
    
    elements = nodes_list + edges_list
    #print(elements)
    return elements

def summary_stats(what_if_sim_run, run_number = 999):
    print(f"What is sim sun is : {what_if_sim_run}")
    nodeData = pd.read_csv("data/cb_node_list.csv", low_memory=False)
    cbEdgeData = pd.read_csv("csv/transition_probs_gp_added.csv", low_memory=False)
    edgeData = pd.read_csv("data/network_graph.csv", low_memory=False, index_col=False) 

    cbdata = cbEdgeData[cbEdgeData['source'] == 6]
    data = edgeData[edgeData['source'] == 6]
    if run_number != 999:
        data = data[data['run_number'] == run_number]

    cbdata2 = pd.merge(cbdata[['target', 'weight']], nodeData, left_on='target', right_on='ID', how='left')
    data2 = pd.merge(data[['target', 'run_number', 'weight']], nodeData, left_on='target', right_on='ID', how='left')
    cbtotal = cbdata2['weight'].sum()
    total = data2['weight'].sum()
    num_runs = data2['run_number'].nunique()

    # total/num_runs
    return_dict = {
        'total': round(total/num_runs)
    }
    cb_return_dict = {
        'cb_total': cbtotal
    }
    wi_return_dict = {}

    if (what_if_sim_run == 'Yes'):
        wi_edgeData = pd.read_csv("data/wi_network_graph.csv", low_memory=False, index_col=False) 
        wi_data = wi_edgeData[wi_edgeData['source'] == 6]
        if run_number != 999:
            wi_data = wi_data[wi_data['run_number'] == run_number]
        wi_data2 = pd.merge(wi_data[['target', 'run_number', 'weight']], nodeData, left_on='target', right_on='ID', how='left')
        wi_total = wi_data2['weight'].sum()
        wi_num_runs = wi_data2['run_number'].nunique()
        wi_return_dict = {
            'wi_total': round(wi_total/wi_num_runs)
        }
        for x in wi_data2['Label'].unique():
            wi_data3 = wi_data2[wi_data2['Label'] == x]
            wi_numerator = wi_data3['weight'].sum()
            wi_prop = round(wi_numerator/wi_total*100,1)
            wi_return_dict[f"wi_{x}"] = wi_prop

    for x in data2['Label'].unique():
        cbdata3 = cbdata2[cbdata2['Label'] == x]
        data3 = data2[data2['Label'] == x]
        cbnumerator = cbdata3['weight'].sum()
        numerator = data3['weight'].sum()
        cbprop = round(cbnumerator/cbtotal*100, 1)
        prop = round(numerator/total*100,1)

        # Timely GP data
        if what_if_sim_run == 'No':
            wi_return_dict[f"wi_{x}"] = None

        cb_return_dict[f"cb_{x}"] = cbprop
        return_dict[x] = prop

    return {**return_dict, **cb_return_dict, **wi_return_dict}


def using_nth(df):
    to_del = df.groupby(['run_number','P_ID'],as_index=False).nth(0)
    return pd.concat([df,to_del]).drop_duplicates(keep=False)


def quarterly_counts(sim_data, sim_data_type = 'Simulation'):

    # Just include start lifecycles
    sim_data2 = sim_data[['pc_outcome', 'activity','timestamp','qtr', 'run_number']][(sim_data['status'] == 'start') & (sim_data['activity'] != 'Index_IUC')]

    sim_data2.timestamp = pd.to_datetime(sim_data2.timestamp)

    sim_data2['quarter'] = sim_data2['qtr']

    sim_data3 = sim_data2.groupby(['run_number','activity', 'quarter'], as_index = False).count()
    sim_data3['count'] = sim_data3['pc_outcome']
    sim_data4 = sim_data3[['run_number','activity', 'quarter', 'count']]

    sim_data5 = sim_data4.groupby(['quarter','activity'], as_index=False)['count'].agg(['mean', 'sem']).reset_index()
    sim_data5['Upper_95_CI'] = round(sim_data5['mean'] + 1.96* sim_data5['sem'],1)
    sim_data5['Lower_95_CI'] = round(sim_data5['mean'] - 1.96* sim_data5['sem'],1)
    sim_data5['data'] = sim_data_type
    sim_data5['count'] = sim_data5['mean'].map(lambda x: round(x)) # Alternative method to round, but not necessary
    sim_data5['mean'] = round(sim_data5['mean'],1)

    #print(pd.concat([sim_data5], axis = 0))

    return pd.concat([sim_data5], axis = 0)


def draw_Sankey(sorted_labels, colorList, short_name, source, target, value):
    fig = go.Figure(data=[go.Sankey(
        valueformat = ".0f",
        arrangement='snap',
        node = {
        "pad": 20,
        "label" : sorted_labels,
        "color" : colorList,
        "customdata": short_name,
        "hovertemplate": '%{customdata} total: %{value}<extra></extra>'
        },
        link = dict(
        source = source,
        target = target,
        value = value,
        hovertemplate='%{source.customdata} to %{target.customdata} total: %{value}<extra></extra>'
    ))])

    fig.update_layout(font_size=14, height=700)

    return fig


def draw_CB_Sankey():
    df = pd.read_csv('./data/cb_sankey.csv')

    nodelist = {
        'Index_IUC' : 'rgba(45, 87, 100, 0.8)',
        'GP': 'rgba(31, 119, 180, 0.8)',
        'IUC': 'rgba(255, 127, 14, 0.8)', 
        '999': 'rgba(44, 160, 44, 0.8)', 
        'ED': 'rgba(214, 39, 40, 0.8)', 
        'IP': 'rgba(148, 103, 189, 0.8)', 
        'End': 'rgba(140, 86, 75, 0.8)'
    }

    uniq_label = pd.unique(df[['source', 'target']].values.ravel('K'))

    sorted_labels = sorted(uniq_label, key=lambda x: 1 if x == 'Index_IUC_1' else int(x.split('_')[1]))
    
    colorList = []
    short_name = []

    for node in sorted_labels:
        nodeName = 'Index_IUC' if node.split('_')[0] == 'Index' else node.split('_')[0]
        a = nodelist[nodeName]
        colorList.append(a)
        short_name.append(nodeName)

    df['source_number'] = df['source'].apply(lambda x: sorted_labels.index(x))
    df['target_number'] = df['target'].apply(lambda x: sorted_labels.index(x))
    df['short_source'] = df['source'].apply(lambda x: x.split('_')[0])
    df['short_target'] = df['target'].apply(lambda x: x.split('_')[0])

    return draw_Sankey(sorted_labels, colorList, short_name, df['source_number'], df['target_number'], df['counts2'])


def prep_draw_Sankey(df, run_number):
    df1 = df[(df['activity'].isin(['End', 'Index_IUC'])) | (df['status'] == 'start')]
    df2 = df1[['P_ID','run_number','activity','instance_id','timestamp']].copy()
    df2['name'] = df2['activity'] + '_' + df2['instance_id'].astype(str).str[:-2]


    df2["source"]=df2.sort_values(['run_number', 'P_ID', 'timestamp']).groupby(['run_number','P_ID'])["name"].shift(1)
    df2['target']=df2.name

    df2a = using_nth(df2)

    df3 = df2a[['source','target']][df2a['source'] != 'End_1'].dropna().value_counts().to_frame('counts').reset_index()

    # https://stackoverflow.com/a/26977495/3650230
    uniq_label = pd.unique(df3[['source', 'target']].values.ravel('K'))

    sorted_labels = sorted(uniq_label, key=lambda x: 1 if x == 'Index_IUC_1' else int(x.split('_')[1]))

    #print(sorted_labels)

    nodelist = {
        'Index_IUC' : 'rgba(45, 87, 100, 0.8)',
        'GP': 'rgba(31, 119, 180, 0.8)',
        'IUC': 'rgba(255, 127, 14, 0.8)', 
        '999': 'rgba(44, 160, 44, 0.8)', 
        'ED': 'rgba(214, 39, 40, 0.8)', 
        'IP': 'rgba(148, 103, 189, 0.8)', 
        'End': 'rgba(140, 86, 75, 0.8)'
    }

    colorList = []
    short_name = []

    for node in sorted_labels:
        nodeName = 'Index_IUC' if node.split('_')[0] == 'Index' else node.split('_')[0]
        a = nodelist[nodeName]
        colorList.append(a)
        short_name.append(nodeName)

    df3['source_number'] = df3['source'].apply(lambda x: sorted_labels.index(x))
    df3['target_number'] = df3['target'].apply(lambda x: sorted_labels.index(x))
    df3['short_source'] = df3['source'].apply(lambda x: x.split('_')[0])
    df3['short_target'] = df3['target'].apply(lambda x: x.split('_')[0])

    if run_number == 999:
        nruns = df['run_number'].nunique()
        df3['counts2'] = df3['counts'].apply(lambda x: round(x/nruns))
    else:
        df3['counts2'] = df3['counts']

    return [sorted_labels, colorList, short_name, df3['source_number'], df3['target_number'], df3['counts2']]

def vol_fig(df, thetitle, run_number, sim = True):

    if sim == False:
        call_act1 = df
    else:
        call_act = (
            df
            .filter(['P_ID', 'run_number', 'day', 'hour'])
            .drop_duplicates()
            .groupby(['run_number','day', 'hour'], as_index=False)
            .count()
        )
        #print(call_act)

        if(run_number < 999):
            # Just return a single run's data

            call_act1 = (
                df[df['run_number'] == run_number]
                .filter(['P_ID', 'run_number', 'day', 'hour'])
                .drop_duplicates()
                .groupby(['run_number','day', 'hour'], as_index=False)['P_ID']
                .count()
            )

        else:
            call_act1 = call_act.groupby(['day','hour'], as_index=True)['P_ID'].agg('mean').reset_index()

    fig = px.histogram(
        call_act1, 
        x='hour', 
        y='P_ID',
        histfunc='avg',
        nbins=100,
        labels={
            'hour': 'Hour of the day',
            'P_ID' : 'Number of patients'
        },
        color='day',
        template='plotly_white',
        category_orders={"day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]},
        color_discrete_sequence= px.colors.sequential.Viridis_r
    )
    fig.layout['yaxis_title'] =  "Number of patients"
    fig.update_traces(marker_line_width=0.5,marker_line_color="#333")
    fig.update_layout(legend_title_text="Day of week")
    fig.update_xaxes(tickvals=list(range(0, 24, 1)))

    return fig
