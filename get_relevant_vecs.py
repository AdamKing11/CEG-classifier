from classifier import load_ceg_nouns


def build_w2v(relevant_tokens, model_file = 'wiki.cy.bin'):
	# using this library because it's more memory friendly for python :)
	from pyfasttext import FastText
	model = FastText(model_file)

	w2v = {}
	for token in relevant_tokens:
		vec = model.get_numpy_vector(token)
		w2v[token] = vec
	return w2v

def save_w2v(w2v, f = 'w2v.cy.json'):
	import json
	w2v = dict((word, vec.tolist()) for word, vec in w2v.items())
	with open(f, 'w') as wf:
		json.dump(w2v, wf)

def load_w2v(f = 'w2v.cy.json'):
	import json
	import numpy as np
	w2v = json.load(open(f))
	w2v = dict((word, np.array(vec)) for word, vec in w2v.items())
	return w2v


if __name__ == '__main__':
	d = load_ceg_nouns()	
	w2v = build_w2v(d)
	save_w2v(w2v)
	print(len(w2v))
	del w2v
	w2v = load_w2v()
	print(len(w2v))