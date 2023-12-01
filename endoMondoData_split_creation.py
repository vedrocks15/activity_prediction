#!/usr/bin/env python
# coding: utf-8

# # Basic Imports

# In[2]:


from pathlib import Path
import os
from tqdm import tqdm_notebook as tqdm
import json
import gzip
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import random
import collections
import pickle
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from haversine import haversine
from math import floor
import pandas as pd
import time, datetime

#import multiprocessing
#from multiprocessing import Pool

import multiprocess as mp


# # Creating Dataset Splits

# In[3]:


# Helper Functions

# Derived features
def calc_mse(avg, hrs):
    dif = (np.array(hrs) - avg) ** 2
    return np.mean(dif)

# converting timestamps to proper format for model consumption
def convert2datetime(unix_timestamp):
    utc_time = time.gmtime(unix_timestamp)
    l = time.localtime(unix_timestamp)
    dt = datetime.datetime(*l[:6])
    return dt

# Function to read data
def process(line):
    return eval(line)

# Z-normalize feature values
def normalize(inp, zMultiple=5):
    mean, std = inp.mean(), inp.std()
    diff = inp - mean
    zScore = diff / std
    return zScore * zMultiple


# In[4]:


# Loading the dataset in the multi-processing fashion.......
path = Path("./")
in_path = str(path / "endomondoHR_proper.json")

# Multiprocessing data read....
pool = mp.Pool(2)
with open(in_path, 'r') as f:
    data = pool.map(process, f)
pool.close()
pool.join()

print("Number of loaded data points :",len(data))


# # More dataset filtering

# In[5]:


data2 = data

# User to workout id mapper.... {user_id_1 : [w_4,w_5,w_8.......]}
user2workout = defaultdict(list)
idxMap = {}
for idx in range(len(data2)):
    d = data2[idx]
    wid = d['id']      # workout id
    uid = d['userId']  # user    id
    user2workout[uid].append(wid)
    
    # all work out ids are different...
    idxMap[wid] = idx

print("Total number of unique users :",len(user2workout)) # how many user


# **Number match as mentioned in the [FitRec](https://sites.google.com/eng.ucsd.edu/fitrec-project/home) site**

# In[6]:


# sorting user's workout based on time for a given workout
for u in user2workout:
    # Workout list...
    workout = user2workout[u]
    
    # Converting timestamp to proper format 
    dts = [(convert2datetime(data2[idxMap[wid]]['timestamp'][0]), wid) for wid in workout]
    dts = sorted(dts, key=lambda x:x[0])
    
    # ascending
    new_workout = [x[1] for x in dts] 
    
    # updating the sorted list
    user2workout[u] = new_workout

# keep 10 core (keeping only the users that have done atleast 10 workouts..)
user2workout_core = defaultdict(list)
for u in user2workout:
    workout = user2workout[u]
    if len(workout) >= 10:
        user2workout_core[u] = workout
        
print("Number of users with alteast 10 workout sessions :",len(user2workout_core))

# total workouts in 10 core subset
count = 0
for u in user2workout_core:
    count += len(user2workout_core[u])
print("Total number of workouts in the 10 session threshold :",count)


# # Mean workout times
# 
# > Global Mean
# 
# > User Mean

# In[8]:


# build time lines (mean times & total time of workouts)
times = defaultdict(float)
user_times = defaultdict(float)

for u in user2workout_core:
    workout = user2workout_core[u]
    tt = []
    
    for wid in workout:
        idx = idxMap[wid]
        d = data2[idx]
        ts = d['timestamp']

        # Keeping track of workout lengths..
        times[wid] = (ts[-1] - ts[0]) / 3600
        tt.append((ts[-1] - ts[0]) / 3600)
    tt = np.array(tt).mean()

    # Usermean time
    user_times[u] = tt

# Global mean time....
vals = np.array(list(times.values()))
print("Print global mean workout time {:.2f} hrs".format(vals.mean()))


# # Previous workouts to current workout
# 
# Each users first workout is removed therefore:
# 
# **1,67,373 = 1,66,417 + 956**

# In[9]:


# contextMap stores all workouts previous to current
contextMap= {}

for u in user2workout_core:
    wids = user2workout_core[u]

    # workout id to index.....
    indices = [idxMap[wid] for wid in wids]

    # build start time
    start_times = []
    for idx in indices:
        start_time = data[idx]['timestamp'][0]
        start_times.append(start_time)

    # build context (0th workout has no i-1 to compare)
    for i in range(1, len(wids)):
        wid = wids[i]
        # Gap between the previous session
        since_last  = (start_times[i] - start_times[i-1]) / (3600*24)
        
        # Gap between the current & first session 
        since_begin = (start_times[i] - start_times[0]) / (3600*24)
        contextMap[wid] = (since_last, since_begin, wids[:i])

print("Size of context map : ",len(contextMap))

since_last_array = []
since_begin_array = []
for wid in contextMap:
    t = contextMap[wid]
    since_last_array.append(t[0])
    since_begin_array.append(t[1])

since_last_array   = np.array(since_last_array)
since_begin_array  = np.array(since_begin_array)

# normalize since last and begin
since_last_array2  = normalize(since_last_array)
since_begin_array2 = normalize(since_begin_array)

# put nomalized since last and begin into contextMap
i = 0
contextMap2 = {}
for wid in contextMap:
    t = contextMap[wid]
    t0 = since_last_array2[i]
    t1 = since_begin_array2[i]
    i += 1
    contextMap2[wid] = (t0, t1, t[2])


# # Final Split of dataset

# In[10]:


# split whole dataset, leave latest into valid and test
train,valid,test = [],[],[]
for u in user2workout_core:
    indices = user2workout_core[u][1:] # remove the first workout since it has no context
    l = len(indices)
    # split in ascending order
    train.extend(indices[:int(0.8*l)])
    valid.extend(indices[int(0.8*l):int(0.9*l)])
    test.extend(indices[int(0.9*l):])

print("Splits : train/valid/test = {}/{}/{}".format(len(train), len(valid), len(test)))


# # Saving as PICKLE

# In[11]:


with open('endomondoHR_proper_temporal_dataset.pkl', 'wb') as f:
    pickle.dump((train,valid,test,contextMap2), f)

