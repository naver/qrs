# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from . import BooleanFeature
from pathlib import Path

import numpy as np

from sentence_transformers import SentenceTransformer, util

class ParaphraseFeature(BooleanFeature):

    def __init__(self, base_sequence, model_name='paraphrase-multilingual-MiniLM-L12-v2', cutoff=0.5, device=None):
        """
        :param base_sequence: the sequence to which to score paraphrases.
        """
        assert cutoff >= 0
        self.model = SentenceTransformer(model_name)
        self.base_sequence = base_sequence
        self.base_emb = self.model.encode([self.base_sequence], convert_to_tensor=True)
        self.cutoff = cutoff
        if device is not None:
            self.model = self.model.to(device)
            self.base_emb = self.base_emb.to(device)
        self.device = device
        super(ParaphraseFeature, self).__init__(
                self.scoring_function,
                "paraphrase")

    def scoring_function(self, x):
        x_emb = self.model.encode([x], convert_to_tensor=True)
        if self.device: self.x_emb = x_emb.to(self.device)
        cosine_similarity = util.pytorch_cos_sim(self.base_emb, x_emb)
        return float(cosine_similarity.cpu().item() > self.cutoff)
