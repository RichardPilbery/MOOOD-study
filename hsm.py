import os
from numpy import NaN
import simpy
import random
import pandas as pd
import numpy as np
from math import floor
from itertools import product
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
     __file__
except NameError: 
    __file__ = sys.argv[0]

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


from g import G
from caller import Caller
from transitions import Transitions

# Health Care System model
# as it relates to 111 patients

class HSM:
    """
        # The health service model class

        This class will simulate the healthcare journey for a caller who has completed NHS pathways triage via the NHS111 service, and has been given a primary care disposition. Callers are 'followed' for 72 hours after triage and contacts with primary care, secondary care, NHS111 and the ambulance service are recorded.

    
    """

    def __init__(self, run_number: int, sim_duration: int, warm_up_duration: int, sim_start_date: str, what_if_sim_run: str, transition_type: str, ia_type: str) :
        print(f"Sim duration inside HSM model is {sim_duration}, transition is {transition_type} and inter_arrival: {ia_type}")
        self.env = simpy.Environment()
        self.patient_counter = 0
        self.sim_duration = sim_duration
        self.warm_up_duration = warm_up_duration
        self.start_dt = sim_start_date
        self.what_if_sim_run = what_if_sim_run

        self.ia_type = ia_type

        self.t = Transitions(transition_type)
        self.transition_type = transition_type
        
        self.GP = simpy.PriorityResource(self.env, capacity=G.number_of_gp)
        self.ED = simpy.PriorityResource(self.env, capacity=G.number_of_ed)
        self.Treble1 = simpy.PriorityResource(self.env, capacity=G.number_of_111)
        self.Treble9 = simpy.PriorityResource(self.env, capacity=G.number_of_999)
        self.IP = simpy.PriorityResource(self.env, G.number_of_IP)
        
        self.mean_q_time_speak_to_gp = 0
        self.mean_q_time_contact_gp = 0

        self.GP_surgery_df =  pd.read_csv("csv/gp_surgeries.csv")
        
        # Create data frame to capture all sim acitivity
        self.results_df                   = pd.DataFrame()
        self.results_df["P_ID"]           = []
        self.results_df["run_number"]     = []
        self.results_df["activity"]       = []
        self.results_df["timestamp"]      = []
        self.results_df["status"]         = []
        self.results_df["instance_id"]    = []
        self.results_df["day"]            = []
        self.results_df["hour"]           = []
        self.results_df["disposition"]    = []
        self.results_df["GP"]             = []
        self.results_df["age"]            = []
        self.results_df["sex"]            = []
        self.results_df["pc_outcome"]     = []
        self.results_df["gp_contact"]     = []
        self.results_df["avoidable"]      = []
        self.results_df.set_index("P_ID", inplace=True)

        # Create a data frame to capture data to create a network diagram with cytoscape
        self.network_df                 = self.create_network_df(run_number)

        self.node_lookup = {
            "999"       : 0,
            "ED"        : 1,
            "End"       : 2,
            "GP"        : 3,
            "IP"        : 4,
            "IUC"       : 5,
            "Index_IUC" : 6
        }
        
        self.run_number = run_number
        
        self.call_interarrival_times_lu = pd.read_csv("csv/inter_arrival_times.csv")

        self.all_results_location = G.all_results_location if what_if_sim_run == 'No' else G.wi_all_results_location
        self.network_graph_location = G.network_graph_location if what_if_sim_run == 'No' else G.wi_network_graph_location


    def create_network_df(self, run_number: int) -> pd.DataFrame:
        """Create an empty pandas dataframe ready to populate with network analysis data

        **Returns:**  
            Pandas dataframe consisting of all possible combination of healthcare system 'nodes'. There are 6 in total. 
            Column names of dataframes are: source, target, run_number, weight  

        """
        seq05 = np.arange(0, 6, 1, dtype=int) # Numbers from 0 to 5, not 6!
        all_combinations = list(product(seq05, seq05))
        index_iuc_combs = list([(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5)])
        df = pd.DataFrame(np.concatenate((all_combinations, index_iuc_combs)), columns=['source','target'])
        df['run_number'] = run_number
        df['weight'] = 0

        return df
        
    # Method to determine the current day, hour and weekday/weekend
    # based on starting day/hour and elapsed sim time
    def date_time_of_call(self, start_dt: str, elapsed_time: int) -> list[int, int, str, int, pd.Timestamp]:
        """
        Calculate a range of time-based parameters given a specific date-time

        **Returns:**  
            list(  
                `dow`             : int  
                `current_hour`    : int  
                `weekday`         : str   
                `current_quarter` : int  
                `current_dt`      : datetime  
            )

        """
        # Elapsed_time = time in minutes since simulation started

        start_dt = pd.to_datetime(start_dt)

        current_dt = start_dt + pd.Timedelta(elapsed_time, unit='min')

        dow = current_dt.strftime('%a')
        # 0 = Monday, 6 = Sunday
        weekday = "weekday" if current_dt.dayofweek < 5 else "weekend"

        current_hour = current_dt.hour

        current_quarter = current_dt.quarter
        
        return [dow, current_hour, weekday, current_quarter, current_dt]
        
    def generate_111_calls(self):
        """
        **Patient generator**
        
        Keeps creating new patients until current time equals sim_duration + warm_up_duration
        
        """
        
        if self.env.now < self.sim_duration + self.warm_up_duration :
            while True:
                self.patient_counter += 1
                
                # Create a new caller/patient
                pt = Caller(self.patient_counter, self.what_if_sim_run)
                
                # Allocate caller to a GP surgery
                pt.gp = random.choices(self.GP_surgery_df["gp_surgery_id"], weights=self.GP_surgery_df["prob"])[0]
                
                # Set caller/patient off on their healthcare journey
                self.env.process(self.patient_journey(pt))
                
                # Get current day of week and hour of day
                [dow, hod, weekday, qtr, current_dt] = self.date_time_of_call(self.start_dt, self.env.now)

                # Update patient instance with time-based values so the current time is known
                pt.day = dow
                pt.hour = hod 
                pt.weekday = weekday
                pt.qtr = qtr

                # This variable is in use, do not comment out!
                weekday_bool = 1 if weekday == 'weekday' else 0
                
                # The interarrival times are recorded in a pandas dataframe.

                # TODO - See if the interarrival times can be improved with machine learning

                # Retrieve mean interarrival time for the specific hour, weekday/weekend and yearly quarter
                inter_time = float(self.call_interarrival_times_lu.query("exit_hour == @hod & weekday == @weekday_bool & qtr == @qtr")["mean_inter_arrival_time"].tolist()[0])

                # Retrieve the maximum interarrival rate for the specific hour, weekday/weekend and yearly quarter
                max_hourly_interarrival_rate = (60 / (float(self.call_interarrival_times_lu.query("exit_hour == @hod & weekday == @weekday_bool & qtr == @qtr")["max_arrival_rate"].tolist()[0])))

                # Retrieve the minimum interarrival rate for the specific hour, weekday/weekend and yearly quarter
                min_hourly_interarrival_rate = (60 / (float(self.call_interarrival_times_lu.query("exit_hour == @hod & weekday == @weekday_bool & qtr == @qtr")["min_arrival_rate"].tolist()[0]))) - 1

                # Determine the interarrival time for the next patient by sampling from the exponential distrubution
                sampled_interarrival = random.expovariate(1.0 / inter_time) 

                # There are a number of user-configurable options to try and improve inter-arrival rate estimation
                # SPOILER: base is the best!

                if self.ia_type == 'base':
                    # Use sampled interarrival time with a check to ensure it does not go over 60 minutes
                    # as this would technically be in the 'next' hour
                    sampled_interarrival = 59 if sampled_interarrival >= 60 else sampled_interarrival
                elif self.ia_type == 'max':
                    # Repeatedly sample until a value that is less than the minimum hourly interarrival rate
                    # is sampled. In tests, this lead to a large drop in overall patient numbers
                    while True if sampled_interarrival <= min_hourly_interarrival_rate else False:
                        sampled_interarrival = random.expovariate(1.0 / inter_time) 
                elif self.ia_type == 'minmax':
                    # Repeatedly sample until a value that is less than the minimum hourly interarrival rate
                    # but greater than the maximum hourly interarrival rate is sampled.
                    # In tests, this also caused a large drop in overall patient numbers
                    while True if max_hourly_interarrival_rate >= sampled_interarrival <= min_hourly_interarrival_rate else False:
                        sampled_interarrival = random.expovariate(1.0 / inter_time) 
                elif self.ia_type == 'mean':
                    # No random sampling at all. Just use the mean interarrival time
                    # Also resulted in lower than expected patient arrivals compared to real life and the 
                    # exponential distribuation sampling. Turns out Dan was right about the exponential distrubtion...
                    sampled_interarrival = inter_time

                # Freeze function until interarrival time has elapsed
                yield self.env.timeout(sampled_interarrival)

    def ooh(self, patient: Caller) -> str:
        """
        Determine whether the patient's index 111 call is in-hours or out-of-hours (ooh)
        
        """
        if patient.weekday != 'weekday':
            return 'ooh'
        elif patient.hour < 8:
            return 'ooh'
        elif patient.hour > 18:
            return 'ooh'
        else:
            return 'in-hours'
            
    def patient_journey(self, patient: Caller) -> None:
        """
            Iterate caller/patient through healthcare system keeping track of elapsed time and
            determine healthcare trajectory
        """

        # Variable to keep track of the order of healthcare interactions.
        # Index_IUC is always 0
        instance_id = 0

        # Capture current simulation time
        patient_enters_sim = self.env.now
    
        # Loop will keep iterating while patient time in sim is less than the time patients are to be followed up
        # i.e. the pt_time_in_sim value (duration in hours)
        while patient.timer < G.pt_time_in_sim:

            # Add boolean to determine whether the patient is still within the simulation warm-up
            # period. If so, then we will not record the patient progress
            not_in_warm_up_period = False if self.env.now < self.warm_up_duration else True
            
            # Increment instance_id
            instance_id += 1

            # Capture current patient actvity
            current_step = patient.activity

            # Determine the patient's next healthcare interaction
            if self.transition_type == 'tree':
                # Determine next step using a decision tree modle
                # SPOILER: it was terrible!

                # Check whether current time is in-hours or out-of-hours
                ooh_check = self.ooh(patient)

                # Use the next_destination_tree function to run the patient through the decision tree model
                next_step = self.t.next_destination_tree(current_step, patient.pc_outcome, patient.gp_timely_callback, patient.qtr, patient.last_step, ooh_check)

                # Before the current step becomes the 'next step', capture current step as previous step
                # in the next iteration....if that makes sense.
                patient.last_step = current_step
            else:
                # Default behaviour
                # Next destination is determined by next_destination function.
                next_step = "GP" if ((self.what_if_sim_run == 'Yes') & (current_step == 'Index_IUC')) else self.t.next_destination(patient.activity, patient.gp_timely_callback, patient.qtr)

            # If next step is ED, we need to determine whether this will be an avoidable admission or not
            ED_urgent = "not applicable"

            # Prepare source and target variables for network dataframe
            source = next((v for k, v in self.node_lookup.items() if k == current_step), None)
            target = next((v for k, v in self.node_lookup.items() if k == next_step), None)

            # Update network graph table
            self.network_df['weight'] = self.network_df.apply(lambda x: x['weight'] + 1 if x['source'] == source and x['target'] == target else x['weight'], axis = 1)


            if (patient.activity == 'Index_IUC') and not_in_warm_up_period:
                # As a one-off, capture the completed Index IUC call
                self.add_patient_result_row('completed', patient, instance_id, ED_urgent)
            
            # Update current patient activity
            patient.activity = next_step
                            
            # Prepare data for event being 'scheduled'
            # The time between this and 'start' will be the queue time
            # or wait time or time for the patient to have another
            # interaction with a healthcare service.
            
            if not_in_warm_up_period:
                self.add_patient_result_row('scheduled', patient, instance_id, ED_urgent)
                
            # Consult the wait_time function to find out how long until the next activity
            wait_time = self.t.wait_time(current_step, patient.activity)
            yield self.env.timeout(wait_time)

            # Based on current patient activity, need to determine where they go next...
            if(patient.activity == 'End'):
                break
            elif patient.activity == 'ED':
                ED_urgent = "non_urgent" if self.t.avoidable_ed_admission(patient.pc_outcome, patient.gp_timely_callback) else "urgent"
                if not_in_warm_up_period:
                    self.add_patient_result_row('start', patient, instance_id, ED_urgent)
                yield self.env.process(self.step_visit(patient, instance_id, ED_urgent))
            else:
                if not_in_warm_up_period:
                    self.add_patient_result_row('start', patient, instance_id, ED_urgent)
                yield self.env.process(self.step_visit(patient, instance_id, ED_urgent))
            
            # Keep track of how long patient has been in sim
            patient.timer = self.env.now - patient_enters_sim


    def step_visit(self, patient: Caller, instance_id: int, ED_urgent: str) -> None:
        """
            Function to retrieve activity time for duration, wait for visit to complete  
            then write result to `result_df`
        """
        
        # Duration of visit
        visit_duration = self.t.activity_time(patient.activity, ED_urgent)

        # Wait until visit is over
        yield self.env.timeout(visit_duration)

        # Update results_df with current values
        self.add_patient_result_row('completed', patient, instance_id, ED_urgent)


    def add_patient_result_row(self, status: str, patient: Caller, instance_id: int, ED_urgent: str) -> None :
        """
            Convenience function to create a row of data for the results table
        
        """
        results = {
            "P_ID"        : patient.id,
            "run_number"  : self.run_number,
            "activity"    : patient.activity,
            "timestamp"   : self.env.now,         
            "status"      : status,
            "instance_id" : instance_id,
            "hour"        : patient.hour,
            "day"         : patient.day,
            "weekday"     : patient.weekday,
            "qtr"         : patient.qtr,
            "GP"          : patient.gp,
            "age"         : patient.age,
            "sex"         : patient.sex,
            "pc_outcome"  : patient.pc_outcome,
            "gp_contact"  : patient.gp_timely_callback,
            "avoidable"   : ED_urgent
        }

        if self.env.now > self.warm_up_duration:
            self.store_patient_results(results)

            
    def store_patient_results(self, results: dict) -> None:      
        """
            Adds a row of data to the Class' `result_df` dataframe
        """

        df_dictionary = pd.DataFrame([results])
        
        self.results_df = pd.concat([self.results_df, df_dictionary], ignore_index=True)   
   
            
    def write_all_results(self) -> None:
        """
            Writes the content of `result_df` to a csv file
        """
        # https://stackoverflow.com/a/30991707/3650230
        
        # Check if file exists...if it does, append data, otherwise create a new file with headers matching
        # the column names of `results_df`
        if not os.path.isfile(self.all_results_location):
           self.results_df.to_csv(self.all_results_location, header='column_names')
        else: # else it exists so append without writing the header
            self.results_df.to_csv(self.all_results_location, mode='a', header=False) 

        # Same as above except this is the network graph data
        if not os.path.isfile(self.network_graph_location):
           self.network_df.to_csv(self.network_graph_location, header='column_names', index=False)
        else: # else it exists so append without writing the header
            self.network_df.to_csv(self.network_graph_location, mode='a', header=False, index=False) 
    
    def run(self) -> None:
        """
            Function to start the simulation. This function is called by the `runSim` function
        """
        # Start entity generators
        self.env.process(self.generate_111_calls())
        
        # Run simulation
        self.env.run(until=(self.sim_duration + self.warm_up_duration + G.pt_time_in_sim))
        
        # Write run results to file
        self.write_all_results() 
        

        
        
        
   