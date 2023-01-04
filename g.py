from scipy.stats import johnsonsu
import numpy as np
# Global parameter values
class G:
    call_inter = 0.5            # Call inter-arrival time in minutes
    start_dt = "2021-01-01"
    
    sim_duration = 5760       # 96 hours 5760 31556952 seconds in a year.
    warm_up_duration = 1440    # 24 hours = 1440
    number_of_runs = 3        # Number of runs
    number_of_gp = 999
    number_of_ed = 999
    number_of_IP = 999
    number_of_111 = 999
    number_of_999 = 999
    
    prob_callback = 0.5
    prob_male = 0.4
    
    pt_time_in_sim = 4320 # 4320 for 72 hours
    
    all_results_location = 'data/all_results.csv'
    network_graph_location = 'data/network_graph.csv'

    wi_all_results_location = 'data/wi_all_results.csv'
    wi_network_graph_location = 'data/wi_network_graph.csv'

    cb_111_call_volumes = 'csv/cb_111_call_volumes.csv'
    cb_node_list = 'csv/cb_node_list.csv'
    cb_sankey_data = 'csv/cb_sankey.csv'
    cb_quarterly_counts = 'csv/cb_quarterly_counts.csv'
    cb_ed_attedance_volumes = 'csv/cb_ed_attendance_volumes.csv'
    cb_ed_attedance_avoidable = 'csv/cb_ed_attendance_avoidable.csv'
    cb_ed_attendance_avoidable_prop = 'csv/cb_ed_attendance_avoidable_prop.csv'

    transition_probs_gp_added = "csv/transition_probs_gp_added.csv"
    transition_probs_gp_added_no_quarters = "csv/transition_probs_gp_added_no_quarters.csv"
    avoidable_admission_by_PC_outcome_GP_contact = "csv/avoidable_admission_by_PC_outcome_GP_contact.csv"
    
    

     