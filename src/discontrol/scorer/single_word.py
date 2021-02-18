# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from . import BooleanFeature


class SingleWordFeature(BooleanFeature):
    def __init__(self, word):
        super(SingleWordFeature, self).__init__(lambda x: word in x, word)
