import pandas as pd
import numpy as np
import os
import sys
import random
import ast
import scipy.stats
import pickle
from g import G

try:
     __file__
except NameError: 
    __file__ = sys.argv[0]

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Class to manage transition probabilities for patients 
# There are differences between those who get a timely GP contact and those who do not, so
# this has been incorporated.

class Transitions:

    def __init__(self, transition_type):
        self.transition_probs_df = pd.read_csv(G.transition_probs_gp_added) if (transition_type == 'qtr') else pd.read_csv(G.transition_probs_gp_added_no_quarters) 
        self.transition_type = transition_type

        # Early (and terrible) attempt at using a decision tree model
        self.dt_model =  pickle.load(open('csv/decision_tree_model.sav', 'rb'))
        
        self.avoidable_ed_admission_df = pd.read_csv(G.avoidable_admission_by_PC_outcome_GP_contact)

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
    
    def next_destination(self, current_step: str, gp_timely_contact: str, quarter: int) -> str:
        """
        Determine the patient's next destination by sampling from previously determined  
        distributions. These were calculated using a custom script coded in R.
        
        """

        # User defined option to use transition probabilities that were calculated taking
        # yearly quarter into account.
        if self.transition_type == 'qtr':
            df = self.transition_probs_df[(self.transition_probs_df.SourceName == current_step) & (self.transition_probs_df.quarter == quarter)]
        else:
            # Ignore yearly quarter adjustment
            df = self.transition_probs_df[(self.transition_probs_df.SourceName == current_step)] 
        
        # There are separate transition probabilities for cases where a timely GP contact was
        # made after the index 111 call
        weights = df.ProbGPYes if gp_timely_contact == "yes" else df.ProbGPNo 

        # Randomly sample a destination from the transition probabilities
        # End as current step should not trigger this function but just in case.
        return "End" if current_step == "End" else random.choices(df.TargetName.tolist(), weights=weights.tolist())[0]

    def wait_time(self, current_activity: str, next_activity: str) -> str :
        """
            Calculate the wait time in seconds before the next activity starts
        """

        # Prepare variable for distribution lookup.
        activity = f"{current_activity}_to_{next_activity}"

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

        return wait_time

    def distribution_finder(self, activity: str, distribution_type : str = 'queue', ED_urgent: str = '') -> list :
        """
            Look up function to retrieve distributions for queue times and activity duration times and
            return them.
        """

        distribution = {}
        return_list = []

        # Select correct list depending on this distribution is being requested
        search_dict = self.queue_time_distr if distribution_type == 'queue' else self.activity_time_distr

        # There are separete distributions for urgent/non-urgent admissions, synonymous with non-avoidable/avoidable
        # admissions

        if activity == 'ED':
            activity = f"{activity}_{ED_urgent}"

        for i in search_dict:
            for k,v in i.items():
                if k == activity:
                    distribution = v
                    break

        if len(distribution) > 0:
            # Found a distribution, return the list
            for k,v in distribution.items():
                return_list.append(k)
                return_list.append(v)
                break

        # TODO: Handle cases where no distribution is found
            
        return return_list

        
    def activity_time(self, activity: str, ED_urgent: str) -> int:
        """
            Look up function to retrieve the appropirate activity time distribution, sample from it and 
            then return the activity time.
        """

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

    def avoidable_ed_admission(self, pc_outcome: str, timely_gp_response: bool) -> bool:
        """
            Function to lookup probability of an admission to ED being avoidable, based on 
            the triage category of the 111 call and whether they saw a GP within the triage
            category timeframe
        """
        # time_gp_response = True/False
        # pc_outcome is string in form contact_2

        aa_prob = self.avoidable_ed_admission_df[(self.avoidable_ed_admission_df['pc_outcome'] == pc_outcome) & (self.avoidable_ed_admission_df['gp_con_time'] == timely_gp_response)]['avoid_att']

        return True if random.uniform(0, 1) < list(aa_prob)[0] else False

    def next_destination_tree(self, current_step: str, pc_outcome: str, gp_timely_contact: str, quarter: int, last_one: str, ooh: str) -> str:
        """
            Function utilising decision tree model to decide on the next activity.
            Currently not in use due to poor performance.
        """
        
        model = pickle.load(open('csv/decision_tree_model.sav', 'rb'))

        df = pd.DataFrame(
            data = [[pc_outcome, current_step, last_one, gp_timely_contact, quarter, ooh]],
            columns = ['pc_outcome', 'previous_one', 'previous_two', 'timely_gp', 'qtr', 'ooh']
        )

        # Data needs to be one_hot_encoded for the model

        df_encoded = pd.get_dummies(df, columns = ['pc_outcome', 'previous_one', 'previous_two', 'timely_gp', 'qtr', 'ooh'])

        # Prepare empty dataframe.

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

        # print(f"Next step is: {next_step}")

        return next_step

