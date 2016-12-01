from __future__ import print_function

import re
from random import shuffle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

def load_CEG(ceg_file):
	mf = []

	with open(ceg_file, "r") as rf:
		for line in rf:
			line = line.rstrip().rsplit("\t")
			if re.search("[mf]", line[1]):
				mf.append((line[0], line[1]))
	return mf 
			
def vectorize(wl, by_char = True):

	char_to_int = {}
	int_to_char = {}
	vl = []

	for w in wl:
		if by_char:
			v = []
			for c in w:
				if c not in char_to_int:
					c_ind = len(char_to_int) + 1
					char_to_int[c] = c_ind
					int_to_char[c_ind] = c
				v.append(char_to_int[c])
			vl.append(v)
			v = []
		else:
			if w not in char_to_int:
				w_ind = len(char_to_int) + 1
				char_to_int[w] = [w_ind]
				int_to_char[w_ind] = w
			vl.append(char_to_int[w])
	
	return vl, int_to_char

def longest_word(l):
	return max(len(w) for w in l)

def format_data(data, split = .75, shuf = False):

	split_index = int(len(data)*split)
	if shuf:
		shuffle(data)

	X = [x[0] for x in data]
	Y = [y[1] for y in data]	# so we just predict the second character, m/f

	x_vec, x_itc = vectorize(X)
	y_vec, y_itc = vectorize(Y, by_char = False)

	max_x_len = longest_word(x_vec)
	max_y_len = longest_word(y_vec)
	
	x_vec = sequence.pad_sequences(x_vec, maxlen=max_x_len)
	y_vec = sequence.pad_sequences(y_vec, maxlen=max_y_len)
	
	x_train = x_vec[:split_index]
	x_test = x_vec[split_index:]
	y_train = y_vec[:split_index]
	y_test = y_vec[split_index:]

	# just to see what a piece of data looks like
	print(data[3])
	print(X[3], Y[3])
	print(x_vec[3], y_vec[3])

	return (x_train, y_train), (x_test, y_test), (max_x_len, max_y_len)


def build_model_1(num_words):
	embed_len = 64
	convo_filters = 32
	filt_len = 3
	pool_len = 3
	lstm_size = 16

	model = Sequential()
	# embedding
	model.add(Embedding(num_words, embed_len, input_length=x_longest))
	# convolution/maxpool
	model.add(Convolution1D(nb_filter=convo_filters, filter_length=filt_len, border_mode='same', activation='relu'))
	model.add(MaxPooling1D(pool_length=pool_len))
	# dropout and LSTM
	model.add(Dropout(0.25))
	model.add(Bidirectional(LSTM(lstm_size))) # unroll to make faster?
	model.add(Dropout(0.25))
	# final layer
	model.add(Dense(y_longest, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model


# gotta beat 74%


data = load_CEG("CEG.NOUNS.txt")
#data = data[:15000]

(x_tr, y_tr), (x_te, y_te), (x_longest, y_longest) = format_data(data, shuf = True)
	
epochs = 5

print("Building model...")
model = build_model_1(num_words = x_tr.shape[0])
print("Done. Now to train!")
#print(model.summary())

model.fit(x_tr, y_tr, nb_epoch=epochs, batch_size=256)

scores = model.evaluate(x_te, y_te, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
