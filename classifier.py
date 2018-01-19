import os, sys, re, csv

import numpy as np
import math, random
from pprint import pprint
from tqdm import tqdm


rand_seed = 1234


def load_ceg_nouns(f = 'LemmaCountsAnalysis.txt'):

	d = {}
	with open(f) as rf:
		
		reader = csv.reader(rf, delimiter = '\t')
		# 0 -> lemma, 1 -> lemma freq, 3 -> lemma POS, 7 -> gender (for nouns)
		# 8 -> token, 9 -> token count 10 -> token freq
		for i, line in enumerate(reader):
			lemma_gender, token, token_freq, token_gender = line[7:11]
			# if not a noun, skip
			if token_gender[0] != 'n':	continue
			# if the token has some weird characters in it, skip (ask mike about weird chars)
			if re.search(r'[^a-z]', token):	continue
			
			if token_gender not in ('nm', 'nf'):	continue
			d[token] = token_gender
			
			#if len(d) > 10:	break
	return d

def prep_data(d, test_split = .8, embed_len = 300):
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
	for i, word in enumerate(all_words[:train_split]):
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

	return (X_tr, y_tr), (X_test, y_test), (c2i, l2i)

def build_phono_model(wordlen, nb_chars, nb_labels):
	from keras.models import Sequential, Model
	from keras.layers import Dense, Input, Masking, Dropout
	from keras.layers import LSTM, Embedding, Bidirectional
	
	char_input = Input((wordlen,))
	masking_layer = Masking(mask_value = 0.)(char_input)
	char_embed_layer = Embedding(nb_chars, 32)(char_input)
	#char_rnn = Bidirectional(LSTM(64, dropout = .5))(char_embed_layer)
	char_rnn = LSTM(64, dropout = .25)(char_embed_layer)
	final_output = Dense(nb_labels, activation = 'softmax')(char_rnn)
	model = Model(char_input, final_output)

	model.compile(loss='categorical_crossentropy', 
		optimizer='rmsprop')

	return model



	

if __name__ == '__main__':
	d = load_ceg_nouns()
	(X_tr, y_tr), (X_test, y_test), (c2i, l2i) = prep_data(d)
	model = build_phono_model(X_tr.shape[1], len(c2i), len(l2i))
	labels = sorted(l2i, key = lambda x : x[1])
	
	from sklearn.metrics import classification_report, accuracy_score
	completed_epochs = 0
	for i in range(15):
		print(i)
		for _ in tqdm(range(15)):
			completed_epochs += 1
			model.fit(X_tr, y_tr, batch_size = 128, epochs = 1, verbose = 0)
		print('scoring....')
		y_hat = np.argmax(model.predict(X_test, verbose = 0), axis=1)
		y_hat_train = np.argmax(model.predict(X_tr, verbose = 0), axis=1)
		print('*' * 40)
		print('\n', classification_report(np.argmax(y_test, axis=1), y_hat, target_names = labels))
		print('test acc:\t', accuracy_score(np.argmax(y_test, axis=1), y_hat))
		print('*' * 20)
		print('train acc:\t', accuracy_score(np.argmax(y_tr, axis=1), y_hat_train))
		print('*' * 80)
		