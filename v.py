import os, sys, re, csv

import numpy as np
import math, random
from pprint import pprint
from tqdm import tqdm


rand_seed = 123

def read_ceg_nouns(f = 'CEG.NOUNS.txt'):
	d = {}
	with open(f) as rf:
		reader = csv.reader(rf, delimiter = '\t')
		for word, label in reader:	
			word = re.sub(r'[^a-z]', '', word.lower())
			# ignore neuter
			if label == 'nd' or word in d:
				continue
			d[word] =  label
	return d

def read_fasttext(f = '/home/ak/Downloads/wiki.cy.vec', word_set = None):
	w2v = {}
	with open(f) as rf:
		next(rf)
		for i, line in enumerate(tqdm(rf)):
			line = line.rstrip().rsplit()
			word = ' '.join(line[:-300]).lower()
			if not word_set or word in word_set:
				vec = np.array(line[-300:], dtype = 'float32')
				w2v[word] = vec 
	return w2v

def prep_data(d, w2v = None, test_split = .8, embed_len = 300):
	chars = set()
	labels = set()
	maxlen = 0
	for word, label in d.copy().items():	
		if len(word) > maxlen: maxlen = len(word)
		for c in word:
			chars.add(c)
		labels.add(label)
	# +1 to leave 0 open for masking
	c2i = dict((c, i + 1) for i, c in enumerate(sorted(chars)))
	l2i = dict((l, i) for i, l in enumerate(sorted(labels)))

	# set up numpy arrays
	# first, figure out the indices
	train_split = int(len(d) * test_split)
	test_split = len(d) - train_split

	X_tr = np.zeros((train_split, maxlen), dtype = 'uint8')
	y_tr = np.zeros((train_split, len(labels)), dtype = 'uint8')
	X_test = np.zeros((test_split, maxlen), dtype = 'uint8')
	y_test = np.zeros((test_split, len(labels)), dtype = 'uint8')

	# so we can control the test/train words
	all_words = sorted(d)
	random.seed(rand_seed)
	random.shuffle(all_words)

	# training
	for i, word in enumerate(all_words[:test_split]):
		label = d[word]
		for j, c in enumerate(word):
			X_tr[i, -len(word) + j] = c2i[c]
		y_tr[i, l2i[label]] = 1 
	
	# testing
	for i, word  in enumerate(all_words[-test_split:]):
		label = d[word]
		for j, c in enumerate(word):
			X_test[i, -len(word) + j] = c2i[c]
		y_test[i, l2i[label]] = 1

	# if we're going to use the word embeddings.....
	if w2v:
		dims = np.sqrt(6) / np.sqrt(embed_len)
		X_vecs_tr = np.zeros((train_split, embed_len), dtype = 'float32')
		X_vecs_test = np.zeros((test_split, embed_len), dtype = 'float32')

		for i, word in enumerate(all_words[:test_split]):
			if word not in w2v:
				#w2v[word] = np.random.uniform(-dims, dims, size = (embed_len,))
				w2v[word] = np.zeros((300), dtype = 'uint8')
			X_vecs_tr[i] = w2v[word]

		for i, word in enumerate(all_words[-test_split:]):
			if word not in w2v:
				#w2v[word] = np.random.uniform(-dims, dims, size = (embed_len,))
				w2v[word] = np.zeros((300), dtype = 'uint8')
			X_vecs_test[i] = w2v[word]

		return ((X_tr, X_vecs_tr), y_tr), ((X_test, X_vecs_test), y_test), (c2i, l2i)	
	else:
		return (X_tr, y_tr), (X_test, y_test), (c2i, l2i)

def build_phono_model(wordlen, nb_chars, nb_labels):
	from keras.models import Sequential, Model
	from keras.layers import Dense, Input, Masking, Dropout
	from keras.layers import LSTM, Embedding
	
	char_input = Input((wordlen,))
	masking_layer = Masking(mask_value = 0.)(char_input)
	char_embed_layer = Embedding(nb_chars, 32)(char_input)
	char_rnn = LSTM(64, dropout = .2)(char_embed_layer)
	final_output = Dense(nb_labels, activation = 'softmax')(char_rnn)
	model = Model(char_input, final_output)

	model.compile(loss='categorical_crossentropy', 
		optimizer='adam')
#		metrics=['accuracy'])
	return model

def build_embedding_model(embded_len, nb_labels):
	from keras.models import Sequential, Model
	from keras.layers import Dense, Input, Masking, Dropout
	from keras.layers import LSTM, Embedding
	
	embed_input = Input((embded_len,))
	embed_hidden_layer = Dense(128, activation = 'relu')(embed_input)
	embed_dropout = Dropout(.2)(embed_hidden_layer)
	final_output = Dense(nb_labels, activation = 'softmax')(embed_dropout)
	
	model = Model(embed_input, final_output)
	model.compile(loss='categorical_crossentropy', 
		optimizer='adam')
	return model

def build_ensemble_model(wordlen, embed_len, nb_chars, nb_labels):
	from keras.models import Sequential, Model
	from keras.layers import Dense, Input, Masking, Dropout
	from keras.layers import LSTM, Embedding, Concatenate

	# phono rnn
	char_input = Input((wordlen,))
	masking_layer = Masking(mask_value = 0.)(char_input)
	char_embed_layer = Embedding(nb_chars, 32)(char_input)
	char_rnn = LSTM(64, dropout = .2)(char_embed_layer)

	# embedding dense
	embed_input = Input((embed_len,))
	embed_hidden_layer = Dense(128, activation = 'relu')(embed_input)
	embed_dropout = Dropout(.2)(embed_hidden_layer)

	# concat the 2 different models
	concat_outputs = Concatenate(axis = -1)([char_rnn, embed_hidden_layer])
	final_output = Dense(nb_labels, activation = 'softmax')(concat_outputs)


	model = Model([char_input, embed_input], final_output)
	model.compile(loss='categorical_crossentropy', 
		optimizer='adam')
	return model	

if __name__ == '__main__':
	from sklearn.metrics import classification_report, accuracy_score

	d = read_ceg_nouns()
	w2v = read_fasttext(word_set = d)
	print(len(w2v), len(d))

	((X_tr, X_vecs_tr), y_tr), ((X_test, X_vecs_test), y_test), (c2i, l2i) = prep_data(d, w2v)			
	labels = sorted(l2i, key = lambda x : x[1])
	print(X_tr.shape, X_test.shape)
	print(X_vecs_tr.shape, X_vecs_test.shape)
	print(y_tr.shape, y_test.shape)
	print(X_tr[0], X_test[0])
	print(y_tr[0], y_test[0])

	if False:
		# ensemble model
		model = build_ensemble_model(X_tr.shape[1], 300, len(c2i), len(l2i))
		for i in range(5):
			print(i)
			for _ in tqdm(range(30)):
				model.fit([X_tr, X_vecs_tr], y_tr, batch_size = 256, epochs = 1, verbose = 0)
			y_hat = np.argmax(model.predict([X_test, X_vecs_test], verbose = 1), axis=1)
			print(classification_report(np.argmax(y_test, axis=1), y_hat, target_names = labels))
			print(accuracy_score(np.argmax(y_test, axis=1), y_hat))
			print('*' * 80)
	elif False:
		# train embedding only model
		model = build_embedding_model(300, len(l2i))
		for i in range(3):
			print(i)
			for _ in tqdm(range(10)):
				model.fit(X_vecs_tr, y_tr, batch_size = 256, epochs = 1, verbose = 0)
			y_hat = np.argmax(model.predict(X_vecs_test, verbose = 1), axis=1)
	
			print(classification_report(np.argmax(y_test, axis=1), y_hat, target_names = labels))
			print(accuracy_score(np.argmax(y_test, axis=1), y_hat))
			print('*' * 80)
	elif True:
		# train phono model
		model = build_phono_model(X_tr.shape[1], len(c2i), len(l2i))

		for i in range(5):
			print(i)
			for _ in tqdm(range(20)):
				model.fit(X_tr, y_tr, batch_size = 128, epochs = 1, verbose = 0)
			y_hat = np.argmax(model.predict(X_test, verbose = 1), axis=1)
	
			print(classification_report(np.argmax(y_test, axis=1), y_hat, target_names = labels))
			print(accuracy_score(np.argmax(y_test, axis=1), y_hat))
			print('*' * 80)
	
	#print(np.argmax(y_hat[:10], axis=1))
	#print(np.argmax(y_test[:10], axis=1))