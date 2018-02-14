# importing modules 
import statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import kurtosis
import csv

# Reading file and saving it in a variable, suppose x;
x = np.array([])    # empty array for storing features
with open('examplecsv.csv') as csvfile:
    f_read = csv.reader(csvfile,delimiter = ',') 
    for i in f_read:
        x = np.append(x,i)

# Extracting PSD features
f, pxx = signal.periodogram(x)
plt.plot(f,pxx);
avg = np.zeros([200])      #200 = number of elements in feature-vector
stn_dev = np.zeros([200])
kurto = np.zeros([200])
step = len(f)/200
for i in range(len(f)/step): 
    avg[i] = np.mean(f[step*i:step*(i+1)])
    stn_dev[i] = statistics.stdev(f[step*i:step*(i+1)]) 
    kurto[i] = kurtosis(f[step*i:step*(i+1)])
    
# concatenate mean, st. deviation and kurtosis
feature_psd = np.concatenate(avg, stn_dev, kurto)