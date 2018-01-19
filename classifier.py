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
			
	return d

def train_and_test_model(model, Xs, ys, test_Xs, test_ys, labels = ('nf', 'nm'), nb_cycles = 15, epochs_per_display = 10):
	from sklearn.metrics import classification_report, accuracy_score
	completed_epochs = 0
	for i in range(nb_cycles):
		print(i * epochs_per_display)
		for _ in tqdm(range(epochs_per_display)):
			completed_epochs += 1
			model.fit(Xs, ys, batch_size = 128, epochs = 1, verbose = 0)
		
		print('scoring....')
		y_hat = np.argmax(model.predict(test_Xs, verbose = 0), axis=1)
		y_hat_train = np.argmax(model.predict(Xs, verbose = 0), axis=1)
		
		print('*' * 20)
		print('\n', classification_report(np.argmax(test_ys, axis=1), y_hat, target_names = labels))
		print('*' * 40)
		print('test acc:\t', accuracy_score(np.argmax(test_ys, axis=1), y_hat))
		print('*' * 60)
		print('train acc:\t', accuracy_score(np.argmax(ys, axis=1), y_hat_train))
		print('*' * 80)

class CEG_nouns(object):

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
		self.X['test_vecs'] = np.zeros((test_split, embed_len), dtype = 'float32')
			
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

def build_phono_model(word_len, nb_chars, nb_labels):
	from keras.models import Model
	from keras.layers import Dense, Input, Dropout
	from keras.layers import LSTM, Embedding, Masking, Bidirectional
	
	char_input = Input((word_len,))
	masking_layer = Masking(mask_value = 0.)(char_input)
	char_embed_layer = Embedding(nb_chars, 32)(masking_layer)
	char_rnn = Bidirectional(LSTM(64, dropout = .25))(char_embed_layer)
	#char_rnn = LSTM(64, dropout = .25)(char_embed_layer)
	final_output = Dense(nb_labels, activation = 'softmax')(char_rnn)
	model = Model(char_input, final_output)

	model.compile(loss='categorical_crossentropy', 
		optimizer='rmsprop')

	return model


def build_embed_model(embed_len, nb_labels):
	from keras.models import Model
	from keras.layers import Dense, Input, Dropout

	embed_input = Input((embed_len,))
	embed_hidden_layer = Dense(128, activation = 'relu')(embed_input)
	embed_dropout = Dropout(.25)(embed_hidden_layer)
	final_output = Dense(nb_labels, activation = 'softmax')(embed_dropout)
	
	model = Model(embed_input, final_output)
	model.compile(loss='categorical_crossentropy', 
		optimizer='rmsprop')
	return model


def build_ensemble_model(embed_len, word_len, nb_chars, nb_labels):
	from keras.models import Model
	from keras.layers import Dense, Input, Dropout, Concatenate
	from keras.layers import LSTM, Embedding, Masking, Bidirectional

	# do the RNN branch of the NN....
	char_input = Input((word_len,))
	masking_layer = Masking(mask_value = 0.)(char_input)
	char_embed_layer = Embedding(nb_chars, 32)(masking_layer)
	char_rnn = LSTM(64, dropout = .25)(char_embed_layer)
	

	#  now the embedding branch.....
	embed_input = Input((embed_len,))
	embed_hidden_layer = Dense(128, activation = 'relu')(embed_input)
	embed_dropout = Dropout(.25)(embed_hidden_layer)

	concat_outputs = Concatenate(axis = -1)([char_rnn, embed_dropout])
	concat_hidden_layer = Dense(64)(concat_outputs)
	concat_dropout = Dropout(.25)(concat_hidden_layer)
	final_output = Dense(nb_labels, activation = 'softmax')(concat_dropout)

	# when we feed this model, we feed it with a list of inputs, e.g. [X_chars, X_embeddings]
	model = Model([char_input, embed_input], final_output)
	model.compile(loss = 'categorical_crossentropy',
		optimizer='rmsprop')
	return model

if __name__ == '__main__':
	from get_relevant_vecs import load_w2v
	nouns = load_ceg_nouns()
	w2v = load_w2v()
	d = CEG_nouns(nouns, w2v)
	
	# phonological model
	model = build_phono_model(d.X['train'].shape[1], len(d.c2i), len(d.l2i))
	train_and_test_model(model, d.X['train'], d.y['train'], d.X['test'], d.y['test'])
	
	# embedding model
	#model = build_embed_model(300, len(d.l2i))
	#train_and_test_model(model, d.X['train_vecs'], d.y['train'], d.X['test_vecs'], d.y['test'])

	# ensemble model
	#model = build_ensemble_model(300, d.X['train'].shape[1], len(d.c2i), len(d.l2i))
	#train_and_test_model(model, 
	#	[d.X['train'], d.X['train_vecs']], d.y['train'],
	#	[d.X['test'], d.X['test_vecs']], d.y['test']) 