# importing required modules
import numpy as np
import statistics
import pywt
import csv
from scipy.stats import kurtosis

# Reading file and saving it in a variable, suppose x;
x = np.array([])    # empty array for storing features
filename = '/Users/aarohitapwal/Desktop/projects/capstone/examplecsv.csv'
with open(filename) as csvfile:
    f_read = csv.reader(csvfile,delimiter = ',') 
    for row in f_read:
        x = np.append(x,[float(j) for j in row])

# wavelet coefficients using wavedec
[a,d1,d2,d3,d4] = pywt.wavedec(x, 'haar', level =4)

#[a,d1,d2,d3,d4] = pywt.wavedec(x, 'db4', level =4) ; Daubechies transform 

def comp_moment(feature):
    '''this function computes the moments like mean, standard deviation 
    and kutosis of the obtained feature vector''' 
    step = int(len(feature)/2)
    # variables to be used inside loops
    avg_temp = np.zeros([2])
    stn_dev_temp = np.zeros([2])
    kurto_temp = np.zeros([2])
    for i in range(int(len(feature)/step)):  
        avg_temp[i] = np.mean(feature[step*i:step*(i+1)])
        stn_dev_temp[i] = statistics.stdev(feature[step*i:step*(i+1)]) 
        kurto_temp[i] = kurtosis(feature[step*i:step*(i+1)])
    return (avg_temp, stn_dev_temp, kurto_temp)
    
# approximation coefficient 
avg_temp, stn_dev_temp, kurto_temp = comp_moment(a)
avg = avg_temp
stn_dev = stn_dev_temp
kurto = kurto_temp

# d1 coffiecient 
avg_temp, stn_dev_temp, kurto_temp = comp_moment(d1)
avg = np.append(avg,avg_temp)
stn_dev = np.append(stn_dev,stn_dev_temp)
kurto = np.append(kurto,kurto_temp)

# d2 coffiecient 
avg_temp, stn_dev_temp, kurto_temp = comp_moment(d2)
avg = np.append(avg, avg_temp)
stn_dev = np.append(stn_dev,stn_dev_temp)
kurto = np.append(kurto,kurto_temp)

# d3 coffiecient 
avg_temp, stn_dev_temp, kurto_temp = comp_moment(d3)
avg = np.append(avg, avg_temp)
stn_dev = np.append(stn_dev,stn_dev_temp)
kurto = np.append(kurto,kurto_temp)

# d4 coffiecient 
avg_temp, stn_dev_temp, kurto_temp = comp_moment(d4)
avg = np.append(avg, avg_temp)
stn_dev = np.append(stn_dev,stn_dev_temp)
kurto = np.append(kurto,kurto_temp)

feature_dwt = np.append(np.append(avg,stn_dev),kurto)