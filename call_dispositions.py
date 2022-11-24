import pandas as pd
import numpy as np
import os
import sys

try:
     __file__
except NameError: 
    __file__ = sys.argv[0]

os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class CallDispositions:
    
    # Note that in the dataset, there were no 20 minute callbacks Dx61
    dx_codes = pd.DataFrame(
        {
            "disposition"       : ["Dx05", "Dx06", "Dx07", "Dx08", "Dx11", "Dx12", "Dx13", "Dx14", "Dx15", "Dx75"],
            "time_frame_hours"  : [2, 6, 12, 24, 1, 2, 6, 12, 24, 72],
            "contact_speak"     : ["contact", "contact", "contact", "contact", "speak", "speak", "speak", "speak", "speak", "contact"],
            "pc_outcome"        : ["contact_2", "contact_6", "contact_12", "contact_24", "speak_1", "speak_2", "speak_6", "speak_12", "speak_24",  "contact_72"],
            "prob"              : [0.296, 0.204, 0.069, 0.125, 0.165, 0.0526, 0.0308, 0.00995, 0.0124, 0.0344],
        }
    )
    dx_codes.set_index("disposition", inplace=True)
    
    hours_of_day = list(range(0,24))
    weekday = ["weekend", "weekday"]

    gp_timely_contact = pd.read_csv("csv/gp_contact_in_timeframe.csv")
    
    