import tensorflow as tf 
import numpy as np 
# import random
 
# # A function to generate a random permutation of arr[]
# def randomize (arr, n):
#     # Start from the last element and swap one by one. We don't
#     # need to run for the first element that's why i > 0
#     for i in range(n-1,0,-1):
#         # Pick a random index from 0 to i
#         j = random.randint(0,i)
 
#         # Swap arr[i] with the element at random index
#         arr[i],arr[j] = arr[j],arr[i]
#     return arr
 
# n = 4 
tf_main = np.load('/input/tf_main.npy')

data = np.array([i[0] for i in tf_main])
label = np.array([i[1] for i in tf_main])

# x_shape = [1280, 60, 41, 2] 
# data = np.random.uniform(size=x_shape)
# label = np.array([randomize([1,0,0,0], 4) for i in range(512)])
# y_shape = label.shape

n = int(len(data)*0.8)
x_train = data[:n]
y_train = label[:n]
x_test = data[n:]
y_test = label[n:]

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


# x = tf.placeholder(tf.float32, shape=[None, len(x_train)])
# x = tf.placeholder(tf.float32)
X_shape = tf_main.shape
X = tf.placeholder(tf.float32, shape=[None, X_shape[0], X_shape[1], X_shape[2]])
Y = tf.placeholder(tf.float32, shape=[None, num_labels])

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

def maxpool2d(x, kernel_size=30, stride_size=[1,2,2,1]):
	return tf.nn.max_pool(x, ksize=[1,kernel_size, kernel_size,1], strides=[1, stride_size, stride_size,1], padding='SAME')

def cnn(x):
	weights = {
	'W_conv1' : tf.Variable(tf.random_normal([kernel_size, kernel_size, num_channels, 32])),
	'W_conv2' : tf.Variable(tf.random_normal([kernel_size, kernel_size, 32, 64])),
	'W_fc' : tf.Variable(tf.random_normal([1024])),
	'out' : tf.Variable(tf.random_normal([1024, num_labels]))
	}

	biases = {
	'b_conv1' : tf.Variable(tf.random_normal([32])),
	'b_conv2' : tf.Variable(tf.random_normal([64])),
	'b_fc' : tf.Variable(tf.random_normal([1024])),
	'out' : tf.Variable(tf.random_normal([num_labels]))
	}

	# x = tf.reshape(x, [-1, bands, frames, num_channels])

	conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
	# conv1 = maxpool2d(conv1)

	conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
	# conv2 = maxpool2d(conv2)

	shape = conv2.get_shape().as_list()

	# print(shape)

	fc = tf.reshape(conv2, [-1, shape[1]*shape[2]*shape[3]])
	# print(fc.shape)
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
	fc = tf.nn.dropout(fc, keep_rate)

	output = tf.matmul(fc, weights['out']) + biases['out']
	return output

def train_cnn(x):
	prediction = cnn(x)
	print(Y)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=label))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	iterations = 10
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for itr in range(iterations):
			offset = (itr*batch_size) % (y_train.shape[0] - batch_size)
			batch_x = x_train[offset:(offset + batch_size), :, :, :]
			batch_y = y_train[offset:(offset + batch_size), :]
			_, c = sess.run([optimizer, cost], feed_dict={X:batch_x , Y:batch_y})
			cost_history = np.append(cost_history, c)
			print("epoch:{}, cost:{}".format(itr, c))
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y,1))
		accuracy = f.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy: ', accuracy.eval({X:x_test, Y: y_test}))

train_cnn(X)
