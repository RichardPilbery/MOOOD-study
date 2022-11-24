"""The health service model"""

import os
from numpy import NaN
import simpy
import csv
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
from call_dispositions import CallDispositions
from caller import Caller
from transitions import Transitions

# Health Care System model
# as it relates to 111 patients



class HSM_Model:
    run_numnber: int
    """Run number (0 if only 1) to keep track of multiple simulation runs"""
    sim_duration: int
    """How long in seconds should the simulation run for?"""
    warm_up_duration: int
    """How long in seconds should the warm-up period last for. Note that no data from warm up is recorded."""
    def __init__(self, run_number, sim_duration, warm_up_duration, sim_start_date, what_if_sim_run, transition_type, ia_type):
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


    def create_network_df(self, run_number) -> pd.DataFrame:
        """Create an empty pandas dataframe ready to populate with network analysis data"""
        seq05 = np.arange(0, 6, 1, dtype=int) # Numbers from 0 to 5, not 6!
        all_combinations = list(product(seq05, seq05))
        index_iuc_combs = list([(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5)])
        df = pd.DataFrame(np.concatenate((all_combinations, index_iuc_combs)), columns=['source','target'])
        df['run_number'] = run_number
        df['weight'] = 0

        return df
        
    # Method to determine the current day, hour and weekday/weekend
    # based on starting day/hour and elapsed sim time
    def date_time_of_call(self, elapsed_time) -> list:
        """
        Calculate a range of time-based parameters given a specific date-time

        Args:
            elapse_time (any): The current date time.

        Returns:
        + dow: int
        + current_hour: int
        + weekday: bool
        + current_quarter: int
        + current_dt: datetime

        """
        # Elapsed_time = time in minutes since simulation started

        start_dt = pd.to_datetime(self.start_dt)

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
        # Run generator until simulation ends
        # Stop creating patients after warmup/sim time to allow existing
        # patients 72 hours to work through sim
        
        if(self.env.now < self.sim_duration + self.warm_up_duration):
            while True:
                self.patient_counter += 1
                
                # Create a new caller
                pt = Caller(self.patient_counter, self.what_if_sim_run)
                #print(f"timely callback is {pt.gp_timely_callback}")
                
                # Allocate them to a GP surgery
                pt.gp = random.choices(self.GP_surgery_df["gp_surgery_id"], weights=self.GP_surgery_df["prob"])[0]
                
                self.env.process(self.patient_journey(pt))
                
                # Get current day of week and hour of day
                [dow, hod, weekday, qtr, current_dt] = self.date_time_of_call(self.env.now)
                pt.day = dow
                pt.hour = hod 
                pt.weekday = weekday

                # This variable is in use, do not comment out!
                weekday_bool = 1 if weekday == 'weekday' else 0
                pt.qtr = qtr
                #print(f"Day is {pt.day} and hour is {pt.hour} and weekday is {pt.weekday} and qtr is {pt.qtr}")
                inter_time = float(self.call_interarrival_times_lu.query("exit_hour == @hod & weekday == @weekday_bool & qtr == @qtr")["mean_inter_arrival_time"].tolist()[0])
                max_hourly_interarrival_rate = (60 / (float(self.call_interarrival_times_lu.query("exit_hour == @hod & weekday == @weekday_bool & qtr == @qtr")["max_arrival_rate"].tolist()[0])))
                min_hourly_interarrival_rate = (60 / (float(self.call_interarrival_times_lu.query("exit_hour == @hod & weekday == @weekday_bool & qtr == @qtr")["min_arrival_rate"].tolist()[0]))) - 1

                sampled_interarrival = random.expovariate(1.0 / inter_time) 
                if self.ia_type == 'base':
                    # Ensure that inter-arrival times cannot be longer than 60 minutes or the lowest freq of
                    # calls in the cBradford 2021 data
                    sampled_interarrival = 59 if sampled_interarrival >= 60 else sampled_interarrival
                elif self.ia_type == 'max':
                    while True if sampled_interarrival <= min_hourly_interarrival_rate else False:
                        sampled_interarrival = random.expovariate(1.0 / inter_time) 
                elif self.ia_type == 'minmax':
                    while True if max_hourly_interarrival_rate >= sampled_interarrival <= min_hourly_interarrival_rate else False:
                        sampled_interarrival = random.expovariate(1.0 / inter_time) 
                elif self.ia_type == 'mean':
                    # No random sampling
                    sampled_interarrival = inter_time

                # Freeze function until interarrival time has elapsed
                yield self.env.timeout(sampled_interarrival)

    def ooh(self, patient):
        if patient.weekday != 'weekday':
            return 'ooh'
        elif patient.hour < 8:
            return 'ooh'
        elif patient.hour > 18:
            return 'ooh'
        else:
            return 'in-hours'
            
    def patient_journey(self, patient):
        # Record the time a patient waits to speak/contact GP
        instance_id = 0
        patient_enters_sim = self.env.now
    
        while patient.timer < (self.warm_up_duration + G.pt_time_in_sim):
            
            instance_id += 1

            current_step = patient.activity

            if self.transition_type == 'tree':
                ooh_check = self.ooh(patient)
                next_step = self.t.next_destination_tree(current_step, patient.pc_outcome, patient.gp_timely_callback, patient.qtr, patient.last_step, ooh_check)
                patient.last_step = current_step
            else:
                next_step = "GP" if ((self.what_if_sim_run == 'Yes') & (current_step == 'Index_IUC')) else self.t.next_destination(patient.activity, patient.gp_timely_callback, patient.qtr) 
            # TODO: Note that technically, we probably should probably calculate the current time and then work out the quarter from that, not
            # when the patient was created. But since they are only around for 72 hours or so, this mostly won't be a problem.

            # If next step is ED, we need to determine whether this will be an avoidable admission or not
            ED_urgent = "not applicable"

            # print(f"Current step is {current_step} and next is {next_step}")
            source = next((v for k, v in self.node_lookup.items() if k == current_step), None)
            target = next((v for k, v in self.node_lookup.items() if k == next_step), None)

            #print(f"Source is {source} and target is {target}")

            # Update network graph table
            self.network_df['weight'] = self.network_df.apply(lambda x: x['weight'] + 1 if x['source'] == source and x['target'] == target else x['weight'], axis = 1)


            if patient.activity == 'Index_IUC':
                # As a one-off, capture Index IUC call
                results = {
                    "patient_id"  : patient.id,
                    "activity"    : patient.activity,
                    "timestamp"   : self.env.now,         
                    "status"      : 'completed',
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
            
            # Update current patient activity
            patient.activity = next_step
                            
            results = {
                "patient_id"  : patient.id,
                "activity"    : patient.activity,
                "timestamp"   : self.env.now,         
                "status"      : 'scheduled',
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
                
            wait_time = self.t.wait_time(current_step, patient.activity)
            yield self.env.timeout(wait_time)

            
            if(patient.activity == 'GP'):
                #print(f'Patient {patient.id} is off to GP')
                with self.GP.request() as req:
                    yield self.env.process(self.step_visit(patient, req, instance_id, 'GP', ED_urgent))
            elif(patient.activity == 'ED'):
                # Need to determine whether the admnission is avoidable (non-urgent) or not (urgent)
                ED_urgent = "non_urgent" if self.t.avoidable_ed_admission(patient.pc_outcome, patient.gp_timely_callback) else "urgent"
                #print(f'Patient {patient.id} is off to ED')
                with self.ED.request() as req:
                    yield self.env.process(self.step_visit(patient, req, instance_id, 'ED', ED_urgent))
            elif(patient.activity == 'IP'):
                #print(f'Patient {patient.id} is off to IP')
                with self.IP.request() as req:
                    yield self.env.process(self.step_visit(patient, req, instance_id, 'IP', ED_urgent))
            elif(patient.activity == 'IUC'):
                #print(f'Patient {patient.id} is off to 111')
                with self.Treble1.request() as req:
                    yield self.env.process(self.step_visit(patient, req, instance_id, 'IUC', ED_urgent))
            elif(patient.activity == '999'):
                #print(f'Patient {patient.id} is off to 999')
                with self.Treble9.request() as req:
                    yield self.env.process(self.step_visit(patient, req, instance_id, '999', ED_urgent))
            elif(patient.activity == 'End'):
                break
            
            patient.timer = self.env.now - patient_enters_sim
        

    def step_visit(self, patient, yieldvalue, instance_id, visit_type, ED_urgent):
        
        # Wait time to access service
        yield yieldvalue
        
        results = {
            "patient_id"  : patient.id,
            "activity"    : visit_type,
            "timestamp"   : self.env.now,         
            "status"      : 'start',
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
        
        # Duration of visit
        visit_duration = self.t.activity_time(visit_type, ED_urgent)
        yield self.env.timeout(visit_duration)
        
        results = {
            "patient_id"  : patient.id,
            "activity"    : visit_type,
            "timestamp"   : self.env.now,         
            "status"      : 'completed',
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
            
    def store_patient_results(self, results):      
        df_to_add = pd.DataFrame(
            {
                "P_ID"            : [results["patient_id"]],
                "run_number"      : [self.run_number],
                "activity"        : [results["activity"]],
                "timestamp"       : [results["timestamp"]],
                "status"          : [results["status"]],
                "instance_id"     : [results["instance_id"]],
                "hour"            : [results["hour"]],
                "day"             : [results["day"]],
                "weekday"         : [results["weekday"]],
                "qtr"             : [results["qtr"]],
                "GP"              : [results["GP"]],
                "age"             : [results["age"]],
                "sex"             : [results["sex"]],
                "pc_outcome"      : [results["pc_outcome"]],
                "gp_contact"      : [results["gp_contact"]],
                "avoidable"       : [results["avoidable"]]
            }
        )
        
        df_to_add.set_index("P_ID", inplace=True)
        self.results_df = self.results_df.append(df_to_add)   
   
            
    def write_all_results(self):
        #print('Writing all results')
        # https://stackoverflow.com/a/30991707/3650230
        
        if not os.path.isfile(self.all_results_location):
           self.results_df.to_csv(self.all_results_location, header='column_names')
        else: # else it exists so append without writing the header
            self.results_df.to_csv(self.all_results_location, mode='a', header=False) 

        if not os.path.isfile(self.network_graph_location):
           self.network_df.to_csv(self.network_graph_location, header='column_names', index=False)
        else: # else it exists so append without writing the header
            self.network_df.to_csv(self.network_graph_location, mode='a', header=False, index=False) 
    
    def run(self):
        # Start entity generators
        self.env.process(self.generate_111_calls())
        
        # Run simulation
        self.env.run(until=(self.sim_duration + self.warm_up_duration + G.pt_time_in_sim))
        
        # Write run results to file
        # self.write_run_results()
        self.write_all_results() 
        

        
        
        
   