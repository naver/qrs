# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from . import BooleanFeature
from pathlib import Path

from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from rake_nltk import Rake
import scipy
import numpy as np

class KeywordsFeature(BooleanFeature):

    def __init__(self, sequence, partial_matches=True, minimal_match=0.5):
        """
        Scores sequences based on whether they contain keywords from an original sequence.
        Keywords are extracted using RAKE. Keywords are not case-sensitive.
        Scores between 0 and 1: 0 when any keyword is not present, 1 only when all keywords are
        exactly present, a score in between when some keywords are only partially present.
        Partial presence means that only some words in a multi-word keyword are in the scored sequence.

        :param sequence: the sequence to extract keywords from.
        """
        rake = Rake()
        rake.extract_keywords_from_text(sequence)
        self.orig = sequence
        tokenizer = TreebankWordTokenizer()
        self.keywords = rake.get_ranked_phrases() if partial_matches else self._process_keywords(rake.get_ranked_phrases())
        self.parts = None
        if partial_matches:
            self.parts = {keyword: tokenizer.tokenize(keyword) for keyword in self.keywords}
        self.partial_matches = partial_matches
        self.minimal_match = minimal_match
        super(KeywordsFeature, self).__init__(
                self.scoring_function,
                "keywords")

    def _process_keywords(self, raw_keywords):
        processed_keywords = []
        tokenizer = TreebankWordTokenizer()
        for keyword in raw_keywords:
            if keyword in self.orig.lower(): 
                processed_keywords.append(keyword) 
            else:
                parts = tokenizer.tokenize(keyword) 
                for part in parts:
                    if part in self.orig.lower():
                        processed_keywords.append(part)
                    else:
                        pass # discard parts that do not occur in the original sequence
        return processed_keywords

    def scoring_function(self, x):
        x = x.lower()
        scores = []
        for keyword in self.keywords:
            if keyword in x:
                scores.append(1.)
            elif self.partial_matches:
                scores.append(self.partial_match(keyword, x))
            else:
                return 0.
        return scipy.stats.mstats.gmean(scores)

    def partial_match(self, keyword, x):
        parts = self.parts[keyword]
        match = np.mean([part in x for part in parts]) 
        return match if match >= self.minimal_match else 0.
