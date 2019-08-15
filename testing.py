import os
import re
import json
import pickle
import numpy as np

from model import *
from postproc.dictionary import Dictionary
from tqdm import *
from data import *
with open('/ssd_scratch/cvit/deep/pickle_files/1601_rand_0.2.preds.pkl', 'rb') as fp:
	data = pickle.load(fp)


class SpellMe(object):
	def __init__(self, savepath):
		self.savepath = savepath
		self.vocab_path = 'lookups/English.vocab.json'
		self._load_vocab()
		nClasses = len(self.lmap)
		nHidden = 512
		self.model = Delayed_LSTM(nClasses, nHidden, nClasses, bidirectional=True).cuda()
		self._load_checkpoint()

	def __call__(self, prediction):
		encoded = torch.Tensor(self._encode(prediction))
		encoded = self._pad(encoded)
		probs = []
		encoded = encoded.unsqueeze(0)
		hidden = self.model.initHidden(1)
		for t in range(encoded.size(1)):
			src = encoded[:, t]
			out, hidden = self.model(src, hidden)
			probs.append(out)
		probs = torch.cat(probs)
		_, indices = torch.max(probs, 1)
		prediction = self._decode(indices)
		prediction = (re.sub('#', '', ''.join(prediction)))
		return prediction

	def _load_checkpoint(self):
		print(f" Loading checkpont")
		checkpoint = torch.load(self.savepath)
		self.model.load_state_dict(checkpoint['state_dict'])

	def _encode(self, word):
	    encoded = list(map(lambda x: self.lmap[x], list(word)))
	    return encoded

	def _decode(self, encoded):
		encoded = encoded.cpu().numpy()
		encoded = list(map(lambda x: self.ilmap[str(x)], list(encoded)))
		return encoded

	def _pad(self, prediction): 
		buffer_, d = 15, 5
		dim = len(prediction)
		padded_pr = torch.ones(dim + 2 * buffer_ +d).long()*89
		padded_pr[buffer_:buffer_+dim] = prediction
		return padded_pr.cuda()

	def _load_vocab(self):
		print('Loading Vocabulary: {}'.format(self.vocab_path))
		with open(self.vocab_path, 'rb') as f:
		    vocab = json.load(f)
		self.lmap, self.ilmap = vocab['v2i'], vocab['i2v']
	
	

obj = SpellMe('saves/Bi_English_error_LSTM.t7')

print(obj('wha'))