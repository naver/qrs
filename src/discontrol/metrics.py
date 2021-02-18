# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

"""
from https://github.com/naver/gdc/blob/master/gdc/gdc/metrics.py
"""
from multiprocessing import Pool
from nltk import ngrams
from nltk.translate.bleu_score import SmoothingFunction

import abc
import nltk
import os

class Metric():
    """
    Defines a text quality metric.
    """

    def get_name(self):
        return self.name

    @abc.abstractmethod
    def compute_metric(self, texts):
        pass


class Distinct_N(Metric):

    def __init__(self, n):
        """
        Distinct n-grams metrics. This is a sequence-level diversity metric.
        See https://www.aclweb.org/anthology/N16-1014 for more details.
        Args:
            n (int): n-grams 
        """

        self.n = n
        self.name = f'Distinct_{n}'

    def compute_metric(self, texts):
        return self._distinct_ngrams(texts, self.n)

    def _distinct_ngrams(self, texts, n):
        total = 0.0
        for t in texts:
            tokens = nltk.tokenize.word_tokenize(t)
            n_distinct = len(set(ngrams(tokens, n)))
            total += n_distinct/ len(tokens)
            # try:
            # except:
            #     continue

        return total / len(texts)


class SelfBLEU(Metric):

    def __init__(self, gram=3, sample_size=500):
        """
        Corpus level diversity metric. See https://arxiv.org/abs/1802.01886 for more details.
        """
        super().__init__()
        self.name = 'Self-BLEU-' + str(gram)
        self.gram = gram
        self.sample_size = sample_size
        self.reference = None
        self.is_first = True

    def compute_metric(self, texts):
        self.reference = texts
        return self._get_bleu_fast()

    def _get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self._get_bleu_fast()
        return self._get_bleu_parallel()

    def _get_reference(self):
        if self.reference is None:
            self.reference = self.test_data
            return self.reference
        else:
            return self.reference

    def _get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self._get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))

        for hypothesis in reference:
            hypothesis = nltk.word_tokenize(hypothesis)
            bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def _calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def _get_bleu_fast(self):
        reference = self._get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self._get_bleu_parallel(reference=reference)

    def _get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self._get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result.append(pool.apply_async(self._calc_bleu, args=(other, hypothesis, weight)))

        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt

