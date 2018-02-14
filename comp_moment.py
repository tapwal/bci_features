import numpy as np
import  statistics
from scipy.stats import kurtosis
def comp_moment(feature):
    '''this function computes the moments like mean, standard deviation 
    and kutosis of the obtained feature vector''' 
    step = len(feature)/5
    # variables to be used inside loops
    avg_temp = np.zeros([5])
    stn_dev_temp = np.zeros([5])
    kurto_temp = np.zeros([5])
    for i in range(len(feature)/step):  
        avg_temp[i] = np.mean(feature[step*i:step*(i+1)])
        stn_dev_temp[i] = statistics.stdev(feature[step*i:step*(i+1)]) 
        kurto_temp[i] = kurtosis(feature[step*i:step*(i+1)])
    return (avg_temp, stn_dev_temp, kurto_temp)