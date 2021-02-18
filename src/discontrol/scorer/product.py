# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from discontrol.scorer import BaseScorer
from functools import reduce

class ProductScorer(BaseScorer):

    def __init__(self, *scorers):
        self.scorers = scorers

    def score(self, items):
        scores = [scorer.score(items) for scorer in self.scorers]
        scores = [scores[0].cast(s) for s in scores]
        return reduce(lambda x,y: x*y, scores)

    def log_score(self, items):
        log_scores = [scorer.log_score(items) for scorer in self.scorers]
        log_scores = [log_scores[0].cast(s) for s in log_scores]
        return reduce(lambda x,y: x+y, log_scores)

    def to(self, device):
        self.scorers = [scorer.to(device) for scorer in self.scorers]
        return self
