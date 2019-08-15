import torch
import numpy as np
from collections import namedtuple
import pdb
from functools import wraps
from time import time as _timenow
from sys import stderr
import os
import pickle
import math
import Levenshtein as lev
from operator import eq
from utils.coding import Decoder
import cv2
import random
import logging
import json

def gmkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


def load(**kwargs):
    base_dir = kwargs['data_dir']
    fname = "%s.data.pkl" % (kwargs["meta"])
    fpath = os.path.join(base_dir, fname)
    if os.path.exists(fpath):
        with open(fpath, 'rb') as fp:
            saved = pickle.load(fp)
            return (saved)
        return (None, False)

class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.total = 0
        self.max = -1*float("inf")
        self.min = float("inf")

    def add(self, element):
        # pdb.set_trace()
        self.total += element
        self.count += 1
        self.max = max(self.max, element)
        self.min = min(self.min, element)

    def compute(self):
        # pdb.set_trace()
        if self.count == 0:
            return float("inf")
        return self.total/self.count

    def __str__(self):
        return "%s (min, avg, max): (%.3lf, %.3lf, %.3lf)" % (self.name, self.min, self.compute(), self.max)


class Eval:
    def __init__(self, vocab):
        self.lmap = vocab['v2i']
        self.ilmap = vocab['i2v']

    def _tostring_(self, indices):
        chars = []
        indices = list(map(str, indices))
        for index in indices:
            chars.append(self.ilmap[index])
        string = ''.join(chars)
        return string

    def batch_compress(self, max_probs):
        def _compress_(values):
            result = []
            for i in range(1, len(values)):
                if values[i] != values[i-1]:
                    result.append(values[i-1])
                    result.append(values[len(values)-1])
            return result
            
        compressed = list(map(_compress_, max_probs))
        return compressed

    def decode_(self, probs):
        """ Convert a probability matrix to sequences """
        # pdb.set_trace()
        # probs = probs.data.squeeze().cpu().numpy()
        # probs = torch.exp(self.normalize(probs))
        # max_vals, max_probs = np.max(probs, axis=1), np.argmax(probs, axis=1)
        max_vals, max_probs = torch.max(probs.transpose(0, 1 ), 2)
        norm_max_vals = torch.exp(self.normalize(max_vals))
        max_probs = max_probs.squeeze().cpu().numpy()
        scores = np.prod(norm_max_vals.cpu().data.numpy(), axis=1)
        ls = max_probs.tolist()
        compressed = self.batch_compress(ls)
        return compressed, scores

    def normalize(self, v):
        # norm = np.linalg.norm(v)
        norm = torch.norm(v)
        if norm == 0: 
           return v
        return v / norm

    def decode(self, probs):
        """ Convert a probability matrix to sequences """
        _, max_probs = torch.max(probs.transpose(0, 1), 2)
        max_probs = max_probs.squeeze().cpu().numpy()
        scores = np.zeros(max_probs.shape[0])
        ls = max_probs.tolist()
        try:
            compressed = self.batch_compress(ls)
        except:
            pdb.set_trace()
        return compressed, scores

    def char_accuracy(self, pair):
        words, truths = pair
        words, truths = ''.join(words), ''.join(truths)
        #pdb.set_trace()
        sum_edit_dists = lev.distance(words, truths)
        sum_gt_lengths = sum(map(len, truths))
        fraction = 0
        if sum_gt_lengths != 0:
            fraction = sum_edit_dists/sum_gt_lengths

        percent = fraction*100
        if 100.0 - percent < 0:
            return 0.0
        else:
            return 100.0 - percent

    def word_accuracy(self, pair):
        correct = 0
        word, truth = pair
        if word == truth:
            correct = 100
        return correct

    def format_target(self, target, target_sizes):
        target_ = []
        start=0
        for size_ in target_sizes:
            target_.append(target[start:start+size_])
            start+=size_
        return target_

    def get_prediction(self, input_):
        input_ ,_ = self.decode(input_)
        return list(map(self._tostring_, input_))
        
    def accuracy(self, input_, target, target_sizes):
        input_, scores = self.decode_(input_)
        target = target.cpu().numpy()
        target = self.format_target(target, target_sizes)
        input_ = list(map(self._tostring_, input_))
        target = list(map(self._tostring_, target))
        pairs = list(zip(input_, target))
        char_acc = np.mean((list(map(self.char_accuracy, pairs))))
        word_acc = np.mean((list(map(self.word_accuracy, pairs))))
        return char_acc, word_acc, scores
