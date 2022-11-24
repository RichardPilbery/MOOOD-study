import pandas as pd
from hsm import HSM

def test_date_time_of_call():
    assert HSM.date_time_of_call('2021-01-01 09:00:00', 60) == ['Fri', 10, 'weekday', 1, pd.Timestamp('2021-01-01 10:00:00')]
    assert HSM.date_time_of_call('2023-05-13 23:59:59', 60) == ['Sun', 0, 'weekend', 2, pd.Timestamp('2023-05-14 00:59:59')]


def test_true_is_false():
    assert True == False