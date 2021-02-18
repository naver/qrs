# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from discontrol.misc import torch_array
from discontrol.sampler import sample_until
from .common import do_batched
from .base import BaseScorer
import torch
import sys
import tqdm


class ExponentialScorer(BaseScorer):
    def __init__(self, features, coefficients):
        self.coefficients = torch_array(coefficients)
        self.features = features

    def to(self, device):
        self.coefficients = self.coefficients.to(device)
        return self

    def score(self, samples):
        return self.log_score(samples).exp()

    def log_score(self, samples):
        feature_scores = [feature.score(samples) for feature in self.features]
        coefficients_T = self.coefficients.cast
        feature_scores = coefficients_T(feature_scores)
        return (self.coefficients * feature_scores.t()).sum(dim=1)


    @classmethod
    def fit(cls, original_distr, features, target_moments, nsamples=4096, iterations=100, lr=5, tolerance=1e-2, batch_size=512, proposal_distr=None):
        if proposal_distr is None:
            proposal_distr = original_distr
        samples = sample_until(proposal_distr, nsamples, silent=False)
        print('Scoring according to q...', file=sys.stderr)
        with torch.no_grad():
            samples_log_q = do_batched(proposal_distr.log_score, batch_size, samples)
        print('Scoring according to a...', file=sys.stderr)
        with torch.no_grad():
            samples_log_a = do_batched(original_distr.log_score, batch_size, samples)
        device = samples_log_q.device
        target_moments = torch_array(target_moments).to(device)
        feature_scores = torch_array([f.score(samples) for f in features]).to(device)
        coefficients = torch_array([0.]*len(features)).to(device)
        assert((feature_scores.min(axis=1).values != feature_scores.max(axis=1).values).all())
        with tqdm.trange(iterations, desc='Fitting exp model') as it:
            for i in it:
                exp = ExponentialScorer(features, coefficients)
                samples_log_P = samples_log_a + exp.log_score(samples).to(device)
                samples_P_over_q = (samples_log_P - samples_log_q).exp()
                moments = (samples_P_over_q * feature_scores).sum(dim=1) / samples_P_over_q.sum()
                grad_coefficients = moments - target_moments
                err = grad_coefficients.abs().max().item()
                it.set_postfix(err=err)
                if err < tolerance:
                    it.total = i
                    it.refresh()
                    break
                coefficients -= lr * grad_coefficients
        return ExponentialScorer(features, coefficients)

    def estimate_moments(self, original_distr, proposal_distr=None, nsamples=4096,
            batch_size=512):
        if proposal_distr is None:
            proposal_distr = original_distr
        samples = sample_until(proposal_distr, nsamples, silent=False)
        samples_log_q = do_batched(proposal_distr.log_score, batch_size, samples)
        device = samples_log_q.device
        samples_log_a = do_batched(original_distr.log_score, batch_size, samples)
        samples_log_P = samples_log_a + self.log_score(samples).to(device)
        samples_P_over_q = (samples_log_P - samples_log_q).exp()
        feature_scores = torch_array([f.score(samples) for f in self.features]).to(device)
        moments = (samples_P_over_q * feature_scores).sum(dim=1) / samples_P_over_q.sum()
        return moments
