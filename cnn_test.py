import os
import tensorflow as tf 
import numpy as np

tf_main = np.load('npy_files/tf_main.npy')

# tf_main = np.load('/home/sound/cnn/2class/npy_files/tf_main.npy')

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(1.0, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

def apply_convolution(x, kernel_size, num_channels, depth):
	weights = weight_variable([kernel_size, kernel_size, num_channels, depth])
	biases = bias_variable([depth])
	return tf.nn.relu(tf.add(conv2d(x, weights), biases))

def apply_max_pool(x, kernel_size, stride_size):
	return tf.nn.max_pool(x, ksize= [1, kernel_size, kernel_size,1], strides=[1, stride_size, stride_size, 1], padding='SAME')


X = np.array([i[0] for i in tf_main])
Y = np.array([i[1] for i in tf_main])

n = int(len(X)*0.8)
x_train = X[:-n]
y_train = Y[:-n]
x_test = X[n:]
y_test = Y[n:]

frames = 41
bands = 60 

feature_size = 2460
num_labels = 4
num_channels = 2

batch_size = 50
kernel_size = 30
depth = 20
num_hidden = 1024
learning_rate = 0.01
training_iterations = 5000



test1 = np.load('npy_files/test/sample1.npy')
test2 = np.load('npy_files/test/sample2.npy')
test3 = np.load('npy_files/test/sample3.npy')
test4 = np.load('npy_files/test/sample4.npy')

tmp = [test1, test2, test3, test4]

X = tf.placeholder(tf.float32, shape=[None, bands, frames, num_channels])
Y = tf.placeholder(tf.float32, shape=[None, num_labels])

cov = apply_convolution(X, kernel_size, num_channels, depth)

shape = cov.get_shape().as_list()
cov_flat = tf.reshape(cov, [-1, shape[1]*shape[2]*shape[3]])


f_weights = weight_variable([shape[1] * shape[2] * depth, num_hidden])
f_biases = bias_variable([num_hidden])
f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights), f_biases))

out_weights = weight_variable([num_hidden, num_labels])
out_biases = bias_variable([num_labels])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

#cost function 
# cost_function = -tf.reduce_sum(Y*tf.log(y_))
cost_function = tf.losses.sigmoid_cross_entropy(Y, y_)
# cost_function = tf.losses.mean_squared_error(Y,y_)

#optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_function)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

#accuracy
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cost_history = np.empty(shape=[1], dtype=float)

#saver 
saver = tf.train.Saver()

with tf.Session() as sess:
	print("+=+=+=+=+=+=+ Test 1 model +=+=+=+=+=+=+=")
	saver.restore(sess, 'models/test1/model.ckpt')
	train_accuracy= round(sess.run(accuracy, feed_dict={X:x_train, Y:y_train}), 3)
	test_accuracy = round(sess.run(accuracy, feed_dict={X:x_test, Y:y_test}), 3)
	print('Train Accuracy: ', train_accuracy )
	print('Test Accuracy: ', test_accuracy )
	for i in range(len(tmp)):
		y_pred = sess.run(y_,feed_dict={X:tmp[i]})
		print("Prediction for sample:{}:{}".format(i,y_pred))
	
	print("+=+=+=+=+=+=+ Test 2 model +=+=+=+=+=+=+=")
	saver.restore(sess, 'models/test2/model.ckpt')
	train_accuracy= round(sess.run(accuracy, feed_dict={X:x_train, Y:y_train}), 3)
	test_accuracy = round(sess.run(accuracy, feed_dict={X:x_test, Y:y_test}), 3)
	print('Train Accuracy: ', train_accuracy )
	print('Test Accuracy: ', test_accuracy )
	for i in range(len(tmp)):
		y_pred = sess.run(y_,feed_dict={X:tmp[i]})
		print("Prediction for sample:{}:{}".format(i,y_pred))

	print("+=+=+=+=+=+=+ Test 3 model +=+=+=+=+=+=+=")
	saver.restore(sess, 'models/test3/model.ckpt')
	train_accuracy= round(sess.run(accuracy, feed_dict={X:x_train, Y:y_train}), 3)
	test_accuracy = round(sess.run(accuracy, feed_dict={X:x_test, Y:y_test}), 3)
	print('Train Accuracy: ', train_accuracy )
	print('Test Accuracy: ', test_accuracy )
	for i in range(len(tmp)):
		y_pred = sess.run(y_,feed_dict={X:tmp[i]})
		print("Prediction for sample:{}:{}".format(i,y_pred))