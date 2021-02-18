# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from . import BaseScorer
from transformers import AutoTokenizer


class ModelFeature(BaseScorer):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def score(self, samples):
        pass        
