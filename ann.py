import tensorflow as tf
import numpy as np 
import librosa
from sklearn.metrics import precision_recall_fscore_support
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.models import Sequential
from keras import losses
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
from random import randint




data = np.load('audio_data_fan.npy')
X_train = data[0:48]
X_test = data[49].reshape(1,193)



#feature extraction



def build_model():
	model = Sequential()
	model.add(Dense(128, input_dim = 193 , activation = 'relu'))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dense(193))
	return model


model = build_model()

checkpoint = ModelCheckpoint('model-{epoch:03d}-{val_loss}.h5',
                                 monitor='val_loss',
                                 verbose=2,
                                 save_best_only=True,
                                 mode='auto')

model.compile(loss = losses.mean_squared_error , optimizer = 'adam')

# def batch_generator(data,batch_size):
# 	init = 0
# 	while True:
# 		X= []
# 		for x in range(init , init + batch_size):
# 			X_i = data[x]
# 			X.append(X_i)
# 		init = init + batch_size
# 		if init == 37:
# 			init = 0
# 		yield np.array(X).reshape(1,193),np.array(X).reshape(1,193)

# model.fit_generator(batch_generator(data,batch_size),samples_per_epoch,nb_epoch,max_q_size=10,validation_data=(X_test, X_test),
#         nb_val_samples=len(X_test),callbacks=[checkpoint],verbose=1)

model.fit(X_train , X_train, batch_size=1, epochs=2500, verbose=1, callbacks=[checkpoint],validation_data=(X_test, X_test))