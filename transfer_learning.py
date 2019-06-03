import numpy as np 
import tensorflow as tf 



tf_main = np.load("npy_files/tf_main_new.npy")
X = [i[0] for i in tf_main]
Y = [i[1] for i in tf_main]


x_train = X[:139]
y_train = Y[:139]
x_test = X[140:]
y_test = Y[140:]
