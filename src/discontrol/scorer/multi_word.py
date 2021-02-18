# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from . import BooleanFeature
from pathlib import Path

class MultiWordFeature(BooleanFeature):
    def __init__(self, words):
        if isinstance(words, str):
            name = Path(words).stem
            with open(words) as f:
                words = [w.rstrip() for w in f]
        else:
            name = "_".join(words)
        super(MultiWordFeature, self).__init__(
                lambda x: any(w in x for w in words),
                name)

