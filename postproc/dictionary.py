from marisa_trie import Trie
import os
from Levenshtein import distance

class Dictionary:
    def __init__(self, *args, **kwargs):
        self.trie = Trie()
        self.secondary_trie = set()
        self.book_trie = Trie()
        path = os.path.dirname(os.path.realpath(__file__)) + "/"
        if 'save' in kwargs:
            self.trie.load(kwargs['save'])
        else:
            with open(kwargs['words'], encoding='utf-8') as fp:
                keys = fp.read().splitlines()
                self.trie = Trie(keys)

        self.preprocess(kwargs['alphabet'])

    def enhance_vocab_with_books(self, words):
        self.book_trie = Trie(words)

    def enhance_vocabulary(self, word):
        self.secondary_trie.add(word)

    def generate_inverse_map(self):
        self.inv_map = {}
        count = 0
        for word in self.trie:
            count = count + 1
            rep = frozenset(list(word))
            if rep not in self.inv_map:
                self.inv_map[rep] = []
            self.inv_map[rep].append(word)
        print(len(self.inv_map.keys()), count)


    def preprocess(self, alphabet_file):
        self.alphabet = open(alphabet_file, encoding='utf-8').read()

    def error(self, word):
        # return (1-int(word in self.trie or word in self.secondary_trie))
        return (1-int(word in self.trie))

    def suggest_v0(self, word):
        rule = lambda x: distance(x, word) <= 3
        suggestions_primary = list(filter(rule, self.trie))
        suggestions_secondary = list(filter(rule, self.secondary_trie))
        suggestions = suggestions_primary + suggestions_secondary
        suggestions = sorted(suggestions, key=lambda x: distance(x, word))
        n = min(10, len(suggestions))
        return suggestions[:10]

    def suggest_v1(self, word):
        # intrie = lambda x: x in self.trie or x in self.secondary_trie or x in self.book_trie
        def intrie(candidate):  
            if (candidate in self.secondary_trie or candidate in self.trie or candidate in self.book_trie): 
                return True
            else: 
                return False
        candidates = list(self.edits1(word) or self.edits2(word))
        in_dictionary = list(filter(intrie, candidates))
        suggestions = sorted(in_dictionary, key=lambda x: distance(x, word))
        n = min(10, len(suggestions))
        return suggestions

    def suggest_v2(self, word):
        rep_word = frozenset(list(word))
        intrie = lambda x: x in self.trie or x in self.secondary_trie or x in self.book_trie
        candidates = []
        for key in self.inv_map:
            if len(rep_word ^ key) < 3:
                candidates.extend(list(filter(intrie, self.inv_map[key])))

        candidates = sorted(list(set(candidates)), key=lambda x: distance(x, word))
        n = min(5, len(candidates))
        print(len(candidates))
        return candidates[:n]

    def suggest(self, word):
        return self.suggest_v1(word)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = self.alphabet 
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))