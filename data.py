import os
import pickle
import json
from tqdm import *
import pdb
from collections import defaultdict
import torch
from torch.utils.data import Dataset
# from . import read_book
# from .lookup import codebook
from noise import noise_maker
from clean import clean_text
# from .transforms import *

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.Tensor(sample)



class WordErrors(Dataset):
    def __init__(self, path, lang, test=False, transform=None):
        super().__init__()
        self.path = path
        self.transform = transform
        self.test = test
        book_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        clean_books = []
        if test:
            with open('../IIIT-H_OCR/data/word_pairs.pkl', 'rb') as fp:
                self.atoms = pickle.load(fp)
        else:
            for book in book_files[:5]:
                book_text = self._load_book_(path + '/' + book)    
                clean_books.append(self._clean_(book_text))
            self.atoms = []
            for book in clean_books:
                for word in book.strip().split(' '):
                    if len(word) < 20:
                        self.atoms.append(word)
        self.vocab_path = os.path.join('lookups', '%s.vocab.json' % lang)
        self._load_vocab()

    def __len__(self):
        return len(self.atoms)

    def __getitem__(self, idx):
        pr, tr = ''.join(self._noise_(self.atoms[idx])), self.atoms[idx]
        pr = torch.Tensor(self._encode_(pr))
        tr = torch.Tensor(self._encode_(tr))
        return pr, tr

    def _noise_(self, word):
        return noise_maker(word, 0.7)

    def _clean_(self, text):
        return clean_text(text)

    def _load_book_(self, path):
        """Load a book from its file"""
        input_file = os.path.join(path)
        with open(input_file) as f:
            book = f.read()
        return book
    
    def _encode_(self, word):
        encoded = list(map(lambda x: self.lmap[x], list(word)))
        return encoded
    
    def _decode_(self, encoded):
        encoded = list(map(lambda x: self.ilmap[x], list(encoded)))
        return encoded

    def _load_vocab(self):
        print('Loading Vocabulary: {}'.format(self.vocab_path))
        with open(self.vocab_path, 'rb') as f:
            vocab = json.load(f)
        self.lmap, self.ilmap = vocab['v2i'], vocab['i2v']

    @staticmethod
    def collate_fn(data):
        """
        data: list ( whatever returned by __getitem___)
        """
        data.sort(key=lambda x: len(x[0]), reverse=True)
        predictions, targets = (zip(*data))
        buffer_, d = 15, 5
        lengths_pr = [len(prediction) for prediction in predictions]
        lengths_tr = [len(target) for target in targets]
        dim = 0
        if max(lengths_pr) > max(lengths_tr):
            dim = max(lengths_pr)
        else:
            dim = max(lengths_tr)
        padded_pr = torch.ones(len(predictions), dim + 2 * buffer_ + d).long()*89

        padded_tr = torch.ones(len(targets),
                               dim + 2 * buffer_ + d).long()*89
        # pdb.set_trace()
        for i, prediction in enumerate(predictions):
            end_pr, end_tr = lengths_pr[i], lengths_tr[i]
            padded_pr[i, buffer_:buffer_+end_pr] = data[i][0]
            padded_tr[i, buffer_+d:buffer_+d+end_tr] = data[i][1]
        return padded_pr, padded_tr, lengths_pr, lengths_tr
        


class DataSeq2Seq(WordErrors):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __len__(self):
        return len(self.atoms)

    def __getitem__(self, idx):
        if self.test:
            pr, tr = self.atoms[idx][0], self.atoms[idx][1]
        else:
            pr, tr = ''.join(self._noise_(self.atoms[idx])), self.atoms[idx]
        pr, tr = self._preprocess_(pr), self._preprocess_(tr)
        pr = torch.Tensor(self._encode_(pr))
        tr = torch.Tensor(self._encode_(tr))
        return pr, tr
    def _preprocess_(self, w):
        return '$' + w + '#'

    @staticmethod
    def collate_fn(data):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        predictions, targets = (zip(*data))
        lengths_pr = [len(prediction) for prediction in predictions]
        lengths_tr = [len(target) for target in targets]
        dim = max(max(lengths_pr), max(lengths_tr))
        padded_pr = torch.ones(len(lengths_pr), dim).long()*58
        padded_tr = torch.ones(len(lengths_tr), dim).long()*58
        for i, prediction in enumerate(predictions):
            padded_pr[i, :lengths_pr[i]] = predictions[i]
            padded_tr[i, :lengths_tr[i]] = targets[i]
        return padded_pr, padded_tr, lengths_pr, lengths_tr