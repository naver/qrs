# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from . import BaseScorer, broadcast

class BooleanFeature(BaseScorer):

    def __init__(self, condition, name):
        self.condition = broadcast(condition)
        self.name = name

    def score(self, items):
        return self.condition(items)
