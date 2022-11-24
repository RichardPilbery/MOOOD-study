from ast import Call
from math import floor
import random
import math
import sys
import os

try:
     __file__
except NameError: 
    __file__ = sys.argv[0]

os.chdir(os.path.dirname(os.path.realpath(__file__)))

from g import G
from call_dispositions import CallDispositions
# Class representing patients who have made a 111 call
class Caller:
    def __init__(self, p_id, what_if_sim_run):
        self.id = p_id
        self.age = math.floor(random.betavariate(0.733, 2.82)*100)
        self.prob_male = G.prob_male
        self.sex = "male" if random.uniform(0, 1) < self.prob_male else "female" 
        self.timer = 0
        self.gp = ""
        self.activity = "Index_IUC" # All callers/patients start with the index 111 call
        # Choose the patient's initial 111 call disposition
        # We can extend this to include symptom groups etc once we have this data
        self.disposition = random.choices(CallDispositions.dx_codes.index, weights=CallDispositions.dx_codes["prob"])[0]
        self.disposition_prob = CallDispositions.dx_codes.prob[CallDispositions.dx_codes.index == self.disposition][0]
        self.pc_outcome = CallDispositions.dx_codes.pc_outcome[CallDispositions.dx_codes.index == self.disposition][0]
        #print(f"PC outcome is {self.pc_outcome}")
        # No idea what is going on here!
        prop = CallDispositions.gp_timely_contact.prop[CallDispositions.gp_timely_contact['pc_outcome'] == self.pc_outcome]
        #print(list(prop)[0])
        self.gp_timely_callback = "yes" if ((random.uniform(0, 1) < list(prop)[0]) | (what_if_sim_run == 'Yes')) else "no"
        
        # Keep track of cumulatative time and exit after 4320 minutes i.e. 72 hours
        self.time_since_call = 0
        # Keep track of time and day caller made call to 111
        # These will all be updated when the patient is created
        self.hour = 0
        self.day = "Mon"
        self.qtr = 1
        self.weekday = "weekday"

        self.last_step = 'None'

    
        
         