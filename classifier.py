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

class CEG_noun_data(object):

	def __init__(self, d, w2v, test_split = .8, embed_len = 300):
		chars = set()
		labels = set()
		maxlen = 0
		for word, label in d.copy().items():
			if len(word) > maxlen: maxlen = len(word)
			for c in word:
				chars.add(c)
			labels.add(label)
		
		# +1 to leave 0 open for masking
		self.c2i = dict((c, i + 1) for i, c in enumerate(sorted(chars)))
		self.l2i = dict((l, i) for i, l in enumerate(sorted(labels)))
	
		# set up numpy arrays
		# first, figure out the indices
		train_split = int(len(d) * test_split)
		test_split = len(d) - train_split
		
		self.X = {}
		self.y = {}
		# arrays for the test data, for strings of characters
		self.X['train'] = np.zeros((train_split, maxlen), dtype = 'uint8')
		self.X['test'] = np.zeros((test_split, maxlen), dtype = 'uint8')
		
		# arrays for word embeddings
		self.X['train_vecs'] = np.zeros((train_split, embed_len), dtype = 'float32')
		self.X['test_vecs'] = np.zeros((train_split, embed_len), dtype = 'float32')
			
		# arrays for labels
		self.y['train'] = np.zeros((train_split, len(labels)), dtype = 'uint8')
		self.y['test'] = np.zeros((test_split, len(labels)), dtype = 'uint8')
	
		# so we can control the test/train words
		all_words = sorted(d)
		random.seed(rand_seed)
		random.shuffle(all_words)
	
		# training
		for i, word in enumerate(all_words[:train_split]):
			label = d[word]
			# assign the word embedding for this word
			self.X['train_vecs'][i, :] = w2v[word]
			# now, make a series of intigers which denote the index for the characters in the word
			# e.g. if the word is "abc" and our self.c2i dictionary is {'a' : 3, 'b' : 1, 'c' : 2},
			# we get "abc" > [3, 1, 2] (with some padding on the right)
			for j, c in enumerate(word):
				self.X['train'][i, -len(word) + j] = self.c2i[c]
			self.y['train'][i, self.l2i[label]] = 1 
			
		# testing
		for i, word  in enumerate(all_words[-test_split:]):
			label = d[word]
			self.X['test_vecs'][i, :] = w2v[word]
			for j, c in enumerate(word):
				self.X['test'][i, -len(word) + j] = self.c2i[c]
			self.y['test'][i, self.l2i[label]] = 1
############

def build_phono_model(wordlen, nb_chars, nb_labels):
	from keras.models import Sequential, Model
	from keras.layers import Dense, Input, Masking, Dropout
	from keras.layers import LSTM, Embedding, Bidirectional
	
	char_input = Input((wordlen,))
	masking_layer = Masking(mask_value = 0.)(char_input)
	char_embed_layer = Embedding(nb_chars, 32)(masking_layer)
	#char_rnn = Bidirectional(LSTM(64, dropout = .5))(char_embed_layer)
	char_rnn = LSTM(64, dropout = .25)(char_embed_layer)
	final_output = Dense(nb_labels, activation = 'softmax')(char_rnn)
	model = Model(char_input, final_output)

	model.compile(loss='categorical_crossentropy', 
		optimizer='rmsprop')

	return model



def build_embed_model(embed_len, nb_labels):
	from keras.models import Sequential, Model
	from keras.layers import Dense, Input, Masking, Dropout
	from keras.layers import LSTM, Embedding

	embed_input = Input((embded_len,))
	embed_hidden_layer = Dense(128, activation = 'relu')(embed_input)
	embed_dropout = Dropout(.25)(embed_hidden_layer)
	final_output = Dense(nb_labels, activation = 'softmax')(embed_dropout)
	
	model = Model(embed_input, final_output)
	model.compile(loss='categorical_crossentropy', 
		optimizer='adam')
	return model

	

def train_and_test_model(model, Xs, ys, test_Xs, test_ys, nb_cycles = 5, epochs_per_display = 10):
	from sklearn.metrics import classification_report, accuracy_score
	completed_epochs = 0
	for i in range(nb_cycles):
		print(i)
		for _ in tqdm(range(epochs_per_display)):
			completed_epochs += 1
			model.fit(Xs, ys, batch_size = 128, epochs = 1, verbose = 0)
		print('scoring....')
		y_hat = np.argmax(model.predict(test_Xs, verbose = 0), axis=1)
		y_hat_train = np.argmax(model.predict(Xs, verbose = 0), axis=1)
		print('*' * 40)
		print('\n', classification_report(np.argmax(test_ys, axis=1), y_hat, target_names = labels))
		print('test acc:\t', accuracy_score(np.argmax(test_ys, axis=1), y_hat))
		print('*' * 20)
		print('train acc:\t', accuracy_score(np.argmax(ys, axis=1), y_hat_train))
		print('*' * 80)


if __name__ == '__main__':
	from get_relevant_vecs import load_w2v
	nouns = load_ceg_nouns()
	w2v = load_w2v()

	#(self.X['tr'], self.y['tr']), (self.X['test'], self.y['test']), (self.c2i, self.l2i) = prep_data(d, w2v)
	d = CEG_noun_data(nouns, w2v)
	labels = sorted(d.l2i, key = lambda x : x[1])
	
	model = build_phono_model(d.X['train'].shape[1], len(d.c2i), len(d.l2i))
	train_and_test_model(model, d.X['train'], d.y['train'], d.X['test'], d.y['test'])
	


	sys.exit()
	from sklearn.metrics import classification_report, accuracy_score
	completed_epochs = 0
	for i in range(15):
		print(i)
		for _ in tqdm(range(5)):
			completed_epochs += 1
			model.fit(d.X['train'], d.y['train'], batch_size = 128, epochs = 1, verbose = 0)
		print('scoring....')
		y_hat = np.argmax(model.predict(d.X['test'], verbose = 0), axis=1)
		y_hat_train = np.argmax(model.predict(d.X['train'], verbose = 0), axis=1)
		print('*' * 40)
		print('\n', classification_report(np.argmax(d.y['test'], axis=1), y_hat, target_names = labels))
		print('test acc:\t', accuracy_score(np.argmax(d.y['test'], axis=1), y_hat))
		print('*' * 20)
		print('train acc:\t', accuracy_score(np.argmax(d.y['train'], axis=1), y_hat_train))
		print('*' * 80)
		