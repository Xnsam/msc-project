import os
import librosa
import numpy as np


def windows(data, window_size):
	start = 0
	while start < len(data):
		yield int(start), int(start + window_size)
		start += (window_size/2)

def extract_features(file_paths, frames = 41, bands = 60):
	window_size = 512 * (frames-1)
	log_specgrams = []
	counter = 0 
	for f in file_paths:
		path = '/home/depasser/xn/sound_prj/sound_project/audio_inputs/'+f
		for files in os.listdir(path):
			path2 = path + '/'+ files
			try:
				# print("entered try")
				counter += 1
				print("extracting file {}:{}".format(f,counter))
				sound_clip , s = librosa.load(path2)
				for i in windows(sound_clip, window_size):
					if(len(sound_clip[i[0]:i[1]]) == window_size):
						signal = sound_clip[i[0]:i[1]]
						melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
						logspec = librosa.logamplitude(melspec)
						logspec = logspec.T.flatten()[:, np.newaxis].T 
						log_specgrams.append(logspec)
			except Exception as e:
				print("Exception:{}, for {}".format(e,f))
	log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames,1)
	features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis= 3)
	for i in range(len(features)):
		features[i,:,:, 1] = librosa.feature.delta(features[i,:,:,0])
	return np.array(features)

file_name1 = ['ac']
train_features_ac = extract_features(file_name1)

file_name2 = ['horn']
train_features_horn = extract_features(file_name2)

np.save('npy_files/train_features_horn.npy',train_features_horn)
np.save('npy_files/train_features_ac.npy',train_features_ac)

labels1 = [1,0,0,0]
labels2 = [0,1,0,0]

tmp =[]
for i in train_features_ac:
	tmp.append([i, labels1])
np.save('npy_files/train_features_ac.npy',tmp)


tmp =[]
for i in train_features_horn:
	tmp.append([i, labels2])
np.save('npy_files/train_features_horn.npy',tmp)


t1 = np.load('npy_files/train_features_horn.npy')
t2 = np.load('npy_files/train_features_ac.npy')

tf_main = np.concatenate((t1,t2), axis=0)

np.random.shuffle(tf_main)
np.save('npy_files/tf_main.npy', tf_main)