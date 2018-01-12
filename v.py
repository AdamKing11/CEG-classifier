import os, sys, re, csv

import numpy as np
import math, random
from pprint import pprint
from tqdm import tqdm



def read_ceg_nouns(f = 'CEG.NOUNS.txt'):
	d = []
	with open(f) as rf:
		reader = csv.reader(rf, delimiter = '\t')
		for word, label in reader:	
			word = re.sub(r'[^a-z]', '', word.lower())
			d.append((word, label))
	return d

def prep_data(d, test_split = .8):
	
	chars = set()
	labels = set()
	maxlen = 0
	for word, label in d:
		if len(word) > maxlen: maxlen = len(word)
		for c in word:
			chars.add(c)
		labels.add(label)

	# +1 to leave 0 open for masking
	c2i = dict((c, i + 1) for i, c in enumerate(sorted(chars)))
	l2i = dict((l, i) for i, l in enumerate(sorted(labels)))

	random.shuffle(d)
	train_split = int(len(d) * test_split)
	test_split = len(d) - train_split

	X_tr = np.zeros((train_split, maxlen), dtype = 'uint8')
	y_tr = np.zeros((train_split), dtype = 'uint8')

	# training
	for i, (word, label) in enumerate(d[:test_split]):
		for j, c in enumerate(word):
			X_tr[i, j] = c2i[c]
		y_tr[i] = l2i[label]
	
	X_test = np.zeros((test_split, maxlen), dtype = 'uint8')
	y_test = np.zeros((test_split), dtype = 'uint8')

	# testing
	for i, (word, label) in enumerate(d[-test_split:]):
		for j, c in enumerate(word):
			X_test[i, j] = c2i[c]
		y_test[i] = l2i[label]

	return (X_tr, y_tr), (X_test, y_test)



if __name__ == '__main__':
	d = read_ceg_nouns()
	(X_tr, y_tr), (X_test, y_test) = prep_data(d)			
	print(X_tr.shape, X_test.shape)
	print(y_tr.shape, y_test.shape)
	print(X_tr[0], X_test[0])
	print(y_tr[0], y_test[0])