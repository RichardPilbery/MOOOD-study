# 111 Primary call disposition
import os
import time
from datetime import datetime
import multiprocessing as mp
import sys
import logging

try:
     __file__
except NameError: 
    __file__ = sys.argv[0]

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from g import G
from hsm import HSM

number_of_runs = G.number_of_runs
warm_up_time = G.warm_up_duration
sim_duration = G.sim_duration
sim_start_date = G.start_dt
what_if_sim_run = "No"
transition_type = "qtr"
ia_type = "base"

def current_time():
    now = datetime.now()
    return now.strftime("%H:%M:%S")

def runSim(run, total_runs, sim_duration, warm_up_time, sim_start_date, what_if_sim_run, transition_type, ia_type):
    #print(f"Inside runSim and {sim_start_date} and {what_if_sim_run}")

    start = time.process_time()

    print (f"{current_time()}: Run {run+1} of {total_runs}")
    logging.debug(f"{current_time()}: Run {run+1} of {total_runs}")
    my_111_model = HSM(run, sim_duration, warm_up_time, sim_start_date, what_if_sim_run, transition_type, ia_type)
    my_111_model.run()
    if what_if_sim_run == 'Yes':
        # If what if model running, also need to run it without to generate comparison results
        print("Starting what if is No simulation following Yes")
        my_111_model = HSM(run, sim_duration, warm_up_time, sim_start_date, "No", transition_type, ia_type)
        my_111_model.run()

    print(f'{current_time()}: Run {run+1} took {round((time.process_time() - start)/60, 1)} minutes to run')
    logging.debug(f'{current_time()}: Run {run+1} took {round((time.process_time() - start)/60, 1)} minutes to run')


def prepStartingVars(**kwargs):
    logging.debug('Prepping starting vars')
    print('Prepping start vars')
    # Update global variables, not create local version with same name
    global number_of_runs
    global warm_up_time
    global sim_duration
    global sim_start_date
    global what_if_sim_run
    global transition_type
    global ia_type

    for k,v in kwargs.items():  
        if k == "number_of_runs":
            number_of_runs = v
        elif k == "warm_up_time":
            warm_up_time = v
        elif k == "sim_duration":
            sim_duration = v
            #print(f"Sim duration in prep starting vars is {sim_duration}")
        elif k == "sim_start_date":
            sim_start_date = v
        elif k == 'what_if_sim_run':
            what_if_sim_run = v
        elif k == 'transition_type':
            transition_type = v
        elif k == 'ia_type':
            ia_type = v

    #print(f"Trans is {transition_type} and IA {ia_type}")
    
def parallelProcess(nprocess = mp.cpu_count() - 1):
    logging.debug('Model called')
    global warm_up_time
    global sim_duration
    global what_if_sim_run
    global transition_type
    global ia_type

    file_locs = [G.all_results_location, G.wi_all_results_location, G.network_graph_location, G.wi_network_graph_location]

    [os.remove(x) for x in file_locs if os.path.isfile(x)]

    pool = mp.Pool(processes = nprocess)
    pool.starmap(runSim, zip(list(range(0, number_of_runs)), [number_of_runs] * number_of_runs, [sim_duration] * number_of_runs, [warm_up_time] * number_of_runs, [sim_start_date] * number_of_runs, [what_if_sim_run] * number_of_runs, [transition_type] * number_of_runs, [ia_type] * number_of_runs))

    logging.debug('Reached end of script')
    logging.shutdown()


#runSim(0, 1, 3, 1, "2021-01-01 00:00:00","Yes", 'base', 'minmax')

if __name__ == "__main__":  
    parallelProcess(nprocess = mp.cpu_count() - 1)
