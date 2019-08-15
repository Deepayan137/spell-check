import torch
import pdb
import numpy as np
from math import log

class Decoder:
    def __init__(self, lmap, ilmap):
        self.lmap = lmap
        self.ilmap = ilmap

    def to_string(self, indices):
        chars = []
        indices = list(map(str, indices))
        for index in indices:
            chars.append(self.ilmap[index])
        string = ''.join(chars)
        return string

    def decode_v1(self, probs):
        score = 0
        _, max_probs = torch.max(probs.transpose(0, 1), 2)
        max_probs = max_probs.data.squeeze().cpu().numpy()
        ls = max_probs.tolist()
        compressed = self.compress(ls)
        return self.to_string(compressed), score


    def decode(self, probs):
        """ Convert a probability matrix to sequences """
        probs = probs.data.squeeze().cpu().numpy()
        probs = np.exp(self.normalize(probs))
        max_vals, max_probs = np.max(probs, axis=1), np.argmax(probs, axis=1)
        score = np.prod(max_vals)
        ls = max_probs.tolist()
        compressed = self.compress(ls)
        return self.to_string(compressed), score

    # def batch_compress(self, max_probs):
    #     def _compress_(values):
    #         result = []
    #         for i in range(1, len(values)):
    #             if values[i] != values[i-1]:
    #                 result.append(values[i-1])
    #                 result.append(values[len(values)-1])
    #         return self.to_string(result)
    #     compressed = list(map(_compress_, max_probs))
    #     return compressed

    def compress(self, values):
        result = []
        for i in range(1, len(values)):
            if values[i] != values[i-1]:
                result.append(values[i-1])
        result.append(values[len(values)-1])
        return result

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0: 
           return v
        return v / norm

    def beam_search_decoder(self, probs, k):
        sequences = [[list(), 1.0]]
        probs = probs.data.squeeze().cpu().numpy()
        probs = np.exp(self.normalize(probs))
        # walk over each step in sequence
        for row in probs:
            all_candidates = list()
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], (score * row[j])]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse=True)
            # select k best
            sequences = ordered[:k]
        strings, scores = [], []
        for i in range(len(sequences)):
            seq, score = sequences[i]
            compressed = self.compress(seq)
            strings.append(self.to_string(compressed))
            scores.append(score)
        return strings