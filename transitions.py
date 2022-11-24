import pandas as pd
import numpy as np
import os
import sys
import random
import ast
import scipy.stats
import pickle

try:
     __file__
except NameError: 
    __file__ = sys.argv[0]

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Class to manage transition probabilities for patients 
# There are differences between those who get a timely GP contact and those who do not, so
# this has been incorporated

class Transitions:

    def __init__(self, transition_type):
        self.transition_probs_df = pd.read_csv("csv/transition_probs_gp_added.csv") if (transition_type == 'qtr') else pd.read_csv("csv/transition_probs_gp_added_no_quarters.csv") 
        self.transition_type = transition_type

        self.dt_model =  pickle.load(open('csv/decision_tree_model.sav', 'rb'))
        
        self.avoidable_ed_admission_df = pd.read_csv("csv/avoidable_admission_by_PC_outcome_GP_contact.csv")
        q_data = []
        with open("csv/queue_time_distributions.txt", "r") as inFile:
            q_data = ast.literal_eval(inFile.read())
        self.queue_time_distr = q_data

        a_data = []
        with open("csv/activity_time_distributions.txt", "r") as inFile:
            a_data = ast.literal_eval(inFile.read())
        self.activity_time_distr = a_data

        inter_arrival_data = []
        with open("csv/inter_arrival_time_distributions.txt", "r") as inFile:
            inter_arrival_data = ast.literal_eval(inFile.read())
        self.inter_arrival_time_distr = inter_arrival_data
    
    def next_destination(self, current_step, gp_timely_contact, quarter):
        if self.transition_type == 'qtr':
            df = self.transition_probs_df[(self.transition_probs_df.SourceName == current_step) & (self.transition_probs_df.quarter == quarter)]
        else:
           df = self.transition_probs_df[(self.transition_probs_df.SourceName == current_step)] 
        
        weights = df.ProbGPYes if gp_timely_contact == "yes" else df.ProbGPNo 
        random.choices(df.TargetName.tolist(), weights=weights.tolist())[0]

        # End as current step should not trigger this function but just in case.
        return "End" if current_step == "End" else random.choices(df.TargetName.tolist(), weights=weights.tolist())[0]

    def wait_time(self, current_activity, next_activity):

        activity = f"{current_activity}_to_{next_activity}"
        #print('Checking ' + activity)

        # Next step is End, so wait time can be 0
        if next_activity == "End":
            # print("End")
            return 0

        wait_time = 0

        distr = self.distribution_finder(activity)

        # Use the appropriate distribution function from scipy.stats
        sci_distr = getattr(scipy.stats, distr[0])
        
        while True:
            # Use the dictionary values, identify them as kwargs for use by scipy distribution function
            q_time = np.floor(sci_distr.rvs(**distr[1]))
            if q_time > 0:
                # Sometimes, a negative number can crop up, which is nonsense with respect to q times.
                wait_time = q_time
                break

        #print(f"Distribution is {distr[0]} and Q time: {q_time}")
        return wait_time

    def distribution_finder(self, activity, distribution_type = 'queue', ED_urgent = ''):
        #print(f"Distribution finder has been sent {activity} and {distribution_type} and {ED_urgent}")
        distribution = {}
        return_list = []

        search_dict = self.queue_time_distr if distribution_type == 'queue' else self.activity_time_distr

        #print(search_dict)

        if activity == 'ED':
            activity = f"{activity}_{ED_urgent}"

        #print(f"Post ed urgent check: {activity}")

        for i in search_dict:
            for k,v in i.items():
                if k == activity:
                    distribution = v
                    break

        #print(f"Distribution is {distribution}")

        if len(distribution) > 0:
            # There's a result
            for k,v in distribution.items():
                return_list.append(k)
                return_list.append(v)
                break
            
        return return_list

        
    def activity_time(self, activity, ED_urgent):
       # print(f"Checking activity time for {activity} and {ED_urgent}")

        gp_visit_time = 10 # Default gp visit time duration of 10 minutes since no end time in data

        activity_time_min = 0

        if activity == 'GP':
            activity_time_min = gp_visit_time
        else:

            #print('Sending to distribution finder')
            distr = self.distribution_finder(activity, 'activity', ED_urgent)

            # Use the appropriate distribution function from scipy.stats
            sci_distr = getattr(scipy.stats, distr[0])
            
            while True:
                # Use the dictionary values, identify them as kwargs for use by scipy distribution function
                a_time = np.floor(sci_distr.rvs(**distr[1]))
                if a_time > 0:
                    # Sometimes, a negative number can crop up, which is nonsense with respect to q times.
                    activity_time_min = a_time
                    break

        return activity_time_min

    def inter_arrival_time(self, qtr, weekend, exit_hour):
        # NOTE: erlang distributions were replaced with gamma to avoid scikit warning
        #print(f"Interarrival time has  {qtr} and {weekend} and {exit_hour}")
        distribution = {}
        return_list = []

        key = f"{qtr}_{weekend}_{exit_hour}"

        #print(f"Key is : {key}")

        for i in self.inter_arrival_time_distr:
            for k,v in i.items():
                if k == key:
                    distribution = v
                    break

        #print(f"Distribution is {distribution}")

        if len(distribution) > 0:
            # There's a result
            for k,v in distribution.items():
                return_list.append(k)
                return_list.append(v)
                break

       # print(return_list)

        sci_distr = getattr(scipy.stats, return_list[0])
            
        while True:
            # Use the dictionary values, identify them as kwargs for use by scipy distribution function
            i_a_time = np.floor(sci_distr.rvs(**return_list[1]))
            if i_a_time > 0 and i_a_time < 60:
                # Sometimes, a negative number can crop up, which is nonsense with respect to q times.
                return i_a_time


    def avoidable_ed_admission(self, pc_outcome, timely_gp_response):
        # time_gp_response = True/False
        # pc_outcome is string in form contact_2

        aa_prob = self.avoidable_ed_admission_df[(self.avoidable_ed_admission_df['pc_outcome'] == pc_outcome) & (self.avoidable_ed_admission_df['gp_con_time'] == timely_gp_response)]['avoid_att']

        return True if random.uniform(0, 1) < list(aa_prob)[0] else False

    def next_destination_tree(self, current_step, pc_outcome, gp_timely_contact, quarter, last_one, ooh):
        model = pickle.load(open('csv/decision_tree_model.sav', 'rb'))

        #print(f"{current_step}, {pc_outcome}, {gp_timely_contact}, {quarter}, {last_one}, {ooh}")

        df = pd.DataFrame(
            data = [[pc_outcome, current_step, last_one, gp_timely_contact, quarter, ooh]],
            columns = ['pc_outcome', 'previous_one', 'previous_two', 'timely_gp', 'qtr', 'ooh']
        )

        df_encoded = pd.get_dummies(df, columns = ['pc_outcome', 'previous_one', 'previous_two', 'timely_gp', 'qtr', 'ooh'])

        #print(df_encoded)

        dummy_df = pd.DataFrame(data = [], columns=['pc_outcome_contact_12', 'pc_outcome_contact_2',
       'pc_outcome_contact_24', 'pc_outcome_contact_6',
       'pc_outcome_contact_72', 'pc_outcome_speak_1', 'pc_outcome_speak_12',
       'pc_outcome_speak_2', 'pc_outcome_speak_24', 'pc_outcome_speak_6',
       'previous_one_999', 'previous_one_ED', 'previous_one_GP',
       'previous_one_IP', 'previous_one_IUC', 'previous_one_Index_IUC',
       'previous_two_999', 'previous_two_ED', 'previous_two_GP',
       'previous_two_IP', 'previous_two_IUC', 'previous_two_Index_IUC', 'previous_two_None',
       'timely_gp_no', 'timely_gp_yes', 'qtr_1', 'qtr_2', 'qtr_3', 'qtr_4',
       'ooh_in-hours', 'ooh_ooh'])


        df2 = dummy_df.append(df_encoded.squeeze())
        df3 = df2.fillna(0)
        next_step = self.dt_model.predict(df3)[0]

        #print(f"Next step is: {next_step}")

        return next_step

