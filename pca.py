#import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler 
# fetching database
mnist = fetch_mldata('MNIST original')
# storing data into a variable, x
x = mnist.data
print(x[:,1].shape)
x = x[0:10,:]
# print(x[0:1,:])
# standardizing the data
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x) # apply transform to the training/testing data
# apply PCA
#pca = PCA(0.95)  # 95 is the percentage variance to be kept
pca = PCA(n_components=200, svd_solver='full')
pca.fit(x)
x = pca.transform(x)
print(x.shape)
