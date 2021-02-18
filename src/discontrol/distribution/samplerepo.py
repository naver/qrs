# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import numpy as np
import torch
import random

from itertools import cycle

from discontrol.scorer import BaseScorer
from discontrol.misc import ListArray, NumpyArray, TorchArray, torch_array

class SampleRepository(BaseScorer):
    
    def __init__(self, sample_file, log_scores_file, batch_size=1, torch=False, shuffle=False):
        samples = np.load(sample_file, allow_pickle=True)
        log_scores = np.load(log_scores_file, allow_pickle=True)
        self.score_dict = {sample: log_score for (sample, log_score) in zip(samples, log_scores)}
        self.samples = samples #np.load(sample_file, allow_pickle=True)
        self.batch_size = batch_size
        self.size = len(samples)
        self.torch = torch

        if shuffle:
            random.shuffle(samples)

        self.repository = iter(samples)

    def sample(self):
        return ListArray([next(self.repository) for _ in range(self.batch_size)])
        
    def score(self, samples):
        return self.log_score(samples).exp()
    
    def log_score(self, samples):
        scores = [self.score_dict[sample] for sample in samples]
        if self.torch:
            return torch_array(scores)
        else:
            return NumpyArray(np.array(scores))

    def reset(self):
        self.repository = iter(self.samples)
