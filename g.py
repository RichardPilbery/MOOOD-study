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
    
    

     