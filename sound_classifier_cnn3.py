import tensorflow as tf 
import numpy as np 


# loading data 
tf_main = np.load('/input/tf_main.npy')

data = np.array([i[0] for i in tf_main])
label = np.array([i[1] for i in tf_main])

n = int(len(data)*0.8)
x_train = data[:n]
y_train = label[:n]
x_test = data[n:]
y_test = label[n:]


#defining the parameters 

bands = 60
frames = 41

keep_rate = 0.8
batch_size = 128

feature_size = 2460
num_labels = 4
num_channels = 2
depth = 32

kernel_size = 30
stride_size = kernel_size // 2


generations = 500 
eval_every = 5
conv1_features = 25 
conv2_features = 50
max_pool_size1 = 2
max_pool_size2 = 2 

fully_connected_size = 100

# defining the inputs 
x_input_shape = [batch_size, bands , frames, num_channels]

x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_target = tf.placeholder(tf.float32, shape=num_labels)

#declare convolution weights and biases

conv1_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, num_channels, conv1_features], stddev=0.1, dtype=tf.float32))
conv1_bias = tf.Variable(tf.truncated_normal([conv1_features]), dtype=float32)

conv2_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, conv1_features, conv2_features], stddev=0.1, dtype=tf.float32))
conv2_bias = tf.Variable(tf.truncated_normal([conv2_features]), dtype=float32)


#declare fully connected weights and biases


# r = int(input("Enter the number of rows\n"))
# c = int(input("Enter the number of columns\n"))
# m = []
# for i in range(r):
# 	tmp = []
# 	for j in range(c):
# 		n1 = input("enter the element in {}:{}\n".format(i,j))
		# tmp.append(int(n1))
# 		print(tmp)
# 	m.append(tmp)