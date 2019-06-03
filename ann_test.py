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
import pyaudio
import wave

#feature extraction
def extract_feature(filename):
	X, sample_rate = librosa.load(filename)
	stft = np.abs(librosa.stft(X))
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T, axis=0)
	return mfccs, chroma, mel, contrast, tonnetz

def parse_audio_files(file_paths):
	features = np.empty((0,193))
	# print("features:{}".format(features.shape))
	for f in file_paths:
		try:
			mfccs, chroma, mel , contrast, tonnetz = extract_feature(f)
		except Exception as e:
			print(e, f)
			continue
		ext_features = np.hstack([mfccs, chroma, mel, contrast,tonnetz])
		# print("ext_features:{}".format(ext_features.shape))
		features = np.vstack([features, ext_features])
	# print("features:{} final".format(features.shape))
	return np.array(features) 

 
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "file.wav"
 
audio = pyaudio.PyAudio()
 

while True:
	# start Recording
	stream = audio.open(format=FORMAT, channels=CHANNELS,
	                rate=RATE, input=True,
	                frames_per_buffer=CHUNK)
	print("recording...")
	frames = []
	 
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	    data = stream.read(CHUNK)
	    frames.append(data)
	print("finished recording")
	 
	 
	# stop Recording
	stream.stop_stream()
	stream.close()
	audio.terminate()
	 
	waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	waveFile.setnchannels(CHANNELS)
	waveFile.setsampwidth(audio.get_sample_size(FORMAT))
	waveFile.setframerate(RATE)
	waveFile.writeframes(b''.join(frames))
	waveFile.close()


	X = parse_audio_files(['file.wav'])




	def build_model():
		model = Sequential()
		model.add(Dense(128, input_dim = 193 , activation = 'relu'))
		model.add(Dense(64, activation = 'relu'))
		model.add(Dense(193))
		return model


	model = build_model()

	# checkpoint = ModelCheckpoint('model-{epoch:03d}-{val_loss}.h5',
	#                                  monitor='val_loss',
	#                                  verbose=2,
	#                                  save_best_only=True,
	#                                  mode='auto')

	#model.compile(loss = losses.mean_squared_error , optimizer = 'adam')

	model.load_weights('p-model-754-3.160309314727783.h5')

	out = model.predict(X, batch_size = 1)
	print(type(out))
	res = losses.mean_squared_error(K.variable(out), K.variable(X))

	print(K.eval(res))








