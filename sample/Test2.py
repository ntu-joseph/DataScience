import numpy as np
import pandas as pd

import datetime as dt
from pyspark.sql import Row
import pyspark

#from __future__ import print_function
#from __future__ import unicode_literals
#from __future__ import division

#spark

from pyspark import SparkConf,SparkContext
conf= SparkConf().setMaster("local").setAppName("Test")
sc=SparkContext(conf=conf)


print("Import End")

start_ts = dt.datetime.strptime("2017-01-02 01:00:00", '%Y-%m-%d %H:%M:%S')
end_ts = dt.datetime.strptime("2017-01-04 00:30:00", '%Y-%m-%d %H:%M:%S')



def parse_slot(d):
    slot = 3
    week_day = d.weekday()
    
    if d < dt.datetime(d.year, d.month, d.day, 1, 0, 0):
        slot = 3
    elif d < dt.datetime(d.year, d.month, d.day, 9, 0, 0):
        slot = 0
    elif d < dt.datetime(d.year, d.month, d.day, 17, 0, 0):
        slot = 1
    elif d < dt.datetime(d.year, d.month, d.day, 21, 0, 0):
        slot = 2
  
    return slot

def get_dime(d):
    return (d - start_ts).days * 4 + parse_slot(d)

def parse_evt(p):
    evt_time = dt.datetime.strptime(p[4][:19], '%Y-%m-%d %H:%M:%S')
    time_diff = evt_time - start_ts 
    dim = get_dime(evt_time)
    
    r = Row(
        user_id=p[0],
        device_id=p[1],
        session_id=p[2],
        title_id=p[3],
        event_time=evt_time,
        played_duration=float(p[5]),
        action_trigger=p[6],
        platform=p[7],
        episode_number=int(p[8]),
        series_total_episodes_count=int(p[9]),
        internet_connection_type=p[10],
        is_trailer=bool(p[11]),
        month=evt_time.month,
        day=evt_time.day,
        hour=evt_time.hour,
        week = (evt_time - start_ts).days // 7,
        slot=parse_slot(evt_time),
        dime=dim
    )
    return r
data_header = sc.textFile('/train').take(1)[0]






print(parse_slot(end_ts))
print(get_dime(end_ts))
print()
print()
print()
print()
print()
print("Test End")