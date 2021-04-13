# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:13:36 2021

@author: Prem Kumar reddy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import tqdm
import random
import datetime
from sklearn.metrics import mean_squared_log_error

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K


#Reading data and creating features for year, month, day, hour
df10320B = pd.read_csv(r'C:\Users\Prem Kumar reddy\Desktop\testbeddata\VB_0320.csv')

df10320B['time'] = pd.to_datetime(df10320B['time'])
df10320B['year'] = df10320B.time.dt.year
df10320B['month'] = df10320B.time.dt.month
df10320B['day'] = df10320B.time.dt.day
df10320B['hour'] = df10320B.time.dt.hour


#sample data plot
df10320B.iloc[8400:8400+7*48,:].plot(y='data', x='time', figsize=(8,6))


#Autocorrelation plot (10 weeks depth)
timeLags = np.arange(1,10*48*7)
autoCorr = [df10320B.data.autocorr(lag=dt) for dt in timeLags]

plt.figure(figsize=(19,8))
plt.plot(1.0/(48*7)*timeLags, autoCorr);
plt.xlabel('time lag [weeks]'); plt.ylabel('correlation coeff', fontsize=12);


#create weekday feature and compute the mean for weejdays at every hour
df10320B['weekday'] = df10320B.time.dt.weekday
df10320B['weekday_hour'] = df10320B.weekday.astype(str) +' '+ df10320B.hour.astype(str)
df10320B['m_weekday'] = df10320B.weekday_hour.replace(df10320B[:25000].groupby('weekday_hour')['data'].mean().to_dict())

#generator for LSTM
sequence_length = 48

def gen_index(id_df, seq_length, seq_cols):

    data_matrix =  id_df[seq_cols]
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(0, num_elements-seq_length, 1), range(seq_length, num_elements, 1)):
        
        yield data_matrix[stop-sequence_length:stop].values.reshape((-1,len(seq_cols)))

### CREATE AND STANDARDIZE DATA FOR LSTM ### 

cnt, mean = [], []
for sequence in gen_index(df10320B, sequence_length, ['data']):
    cnt.append(sequence)
    
for sequence in gen_index(df10320B, sequence_length, ['m_weekday']):
    mean.append(sequence)

cnt, mean = np.log(cnt), np.log(mean)
cnt = cnt - mean

### CREATE AND STANDARDIZE LABEL FOR LSTM ###

init = df10320B.m_weekday[sequence_length:].apply(np.log).values
label = df10320B.data[sequence_length:].apply(np.log).values - init

### DEFINE QUANTILE LOSS ###

def q_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

## TRAIN TEST SPLIT ###

X_train, X_test = cnt[:30000], cnt[19000:]
y_train, y_test = label[:30000], label[19000:]
train_date, test_date = df10320B.time.values[sequence_length:30000+sequence_length], df10320B.time.values[19000+sequence_length:]


tf.random.set_seed(33)
os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), 
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)


### CREATE MODEL ###

losses = [lambda y,f: q_loss(0.1,y,f), lambda y,f: q_loss(0.5,y,f), lambda y,f: q_loss(0.9,y,f)]

inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
lstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5))(inputs, training = True)
lstm = Bidirectional(LSTM(16, return_sequences=False, dropout=0.5))(lstm, training = True)
dense = Dense(50)(lstm)
out10 = Dense(1)(dense)
out50 = Dense(1)(dense)
out90 = Dense(1)(dense)
model = Model(inputs, [out10,out50,out90])

model.compile(loss=losses, optimizer='adam', loss_weights = [0.3,0.3,0.3])
model.fit(X_train, [y_train,y_train,y_train], epochs=50, batch_size=128, verbose=2)



### QUANTILEs BOOTSTRAPPING ###

pred_10 = []
pred_50 = []
pred_90 = []

for i in tqdm.tqdm(range(0,100)):
    predd = model.predict(X_test)
    pred_10.append(predd[0])
    pred_50.append(predd[1])
    pred_90.append(predd[2])

pred_10 = np.asarray(pred_10)[:,:,0] 
pred_50 = np.asarray(pred_50)[:,:,0]
pred_90 = np.asarray(pred_90)[:,:,0]

### REVERSE TRANSFORM PREDICTIONS ###

pred_90_m = np.exp(np.quantile(pred_90,0.9,axis=0) + init[30000:])
pred_50_m = np.exp(pred_50.mean(axis=0) + init[30000:])
pred_10_m = np.exp(np.quantile(pred_10,0.1,axis=0) + init[30000:])

### EVALUATION METRIC ###

mean_squared_log_error(np.exp(y_test + init[30000:]), pred_50_m)

### PLOT QUANTILE PREDICTIONS ###

plt.figure(figsize=(16,8))
plt.plot(test_date, pred_90_m, color='cyan')
plt.plot(test_date, pred_50_m, color='blue')
plt.plot(test_date, pred_10_m, color='green')


### CROSSOVER CHECK ###

plt.scatter(np.where(np.logical_or(pred_50_m>pred_90_m, pred_50_m<pred_10_m))[0], 
            pred_50_m[np.logical_or(pred_50_m>pred_90_m, pred_50_m<pred_10_m)], c='red', s=50)

### PLOT UNCERTAINTY INTERVAL LENGTH WITH REAL DATA ###

plt.figure(figsize=(16,8))
plt.plot(test_date, np.exp(y_test + init[30000:]), color='red', alpha=0.4)
plt.scatter(test_date, pred_90_m - pred_10_m)





