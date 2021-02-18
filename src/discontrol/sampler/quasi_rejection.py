# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from . import BaseSampler
import numpy as np
import tqdm

class QuasiRejectionSampler(BaseSampler):

    def __init__(self, ebm, proposal, batch_size, min_acceptance_rate=0, silent=False):
        super(QuasiRejectionSampler, self).__init__(ebm, proposal)
        self.batch_size = batch_size
        self.beta = 0        
        self.accepted = 0
        self.total = 0
        self.beta_hist = [self.beta]
        self.min_acceptance_rate = min_acceptance_rate
        self.silent = silent


    def sample(self):
        samples = SamplesAccumulator()
        alpha_pcts = Percentiles()
        if not self.silent:
            prog = tqdm.tqdm(desc="QRS", total=self.batch_size if self.batch_size else 1)
        while self.batch_size is None or len(samples) < self.batch_size:
            x, beta_x = self._sample_and_assess_importance(self.batch_size - len(samples) if self.batch_size else None)
            u = beta_x.new_uniform(0, 1)
            alpha_x = beta_x / u
            alpha_pcts.add(alpha_x)
            max_allowable_beta = alpha_pcts.at(1-self.min_acceptance_rate)
            max_sample_beta = beta_x.maxval()
            beta_proposal = min(max_allowable_beta, max_sample_beta)
            if beta_proposal > self.beta:
                self.beta = beta_proposal
                self.beta_hist.append(self.beta)
                if not samples.empty():
                    n_rejected = int((samples.alphas <= self.beta).sum())
                    self.accepted -= n_rejected
                    samples.filter(samples.alphas > self.beta)
                    if not self.silent:
                        prog.n = self.accepted
                        prog.set_postfix(beta=self.beta, ar=self.get_acceptance_rate())
                        prog.update()
            x = x.filter(alpha_x > self.beta)
            alpha_x = alpha_x.filter(alpha_x > self.beta)
            self.accepted += len(x)
            if not self.silent:
                prog.update(len(x))
                prog.set_postfix(beta=self.beta, ar=self.get_acceptance_rate())
            samples.add(x, alpha_x)
            if self.batch_size is None:
                break
        if not self.silent:
            prog.close()
        return samples.values

    def _sample_and_assess_importance(self, max_qty):
        assert max_qty is None or max_qty > 0
        x = self.proposal.sample()
        if max_qty is not None and len(x) > max_qty:
            x = x[:max_qty]
        self.total += len(x)
        try:
            q_x = self.proposal.log_score(x)
            P_x = self.ebm.log_score(x)
            beta_x = (P_x - q_x).exp()
        except AttributeError:
            q_x = self.proposal.score(x)
            P_x = self.ebm.score(x)
            beta_x = P_x / q_x
        return x, beta_x

    def get_acceptance_rate(self):
        return self.accepted / self.total

    def get_tvd_bound(self, delta=0.05):
        return np.sqrt(self.beta ** 2 / self.total / 2 * (-np.log(delta)))

    def get_meta(self):
        return {
                'batch_size': self.batch_size,
                'beta': self.beta,
                'beta_hist': self.beta_hist,
                'tvd_bound:d=0.05': self.get_tvd_bound(delta=0.05),
                'accepted': self.accepted,
                'total': self.total}


class SamplesAccumulator(object):
    def __init__(self, values=None, alphas=None):
        self.values = values
        self.alphas = alphas

    def empty(self):
        return self.values is None

    def add(self, values, alphas):
        self.values = values.concat_to(self.values)
        self.alphas = alphas.concat_to(self.alphas)

    def filter(self, cond):
        self.values = self.values.filter(cond)
        self.alphas = self.alphas.filter(cond)

    def __len__(self):
        return len(self.values) if self.values else 0

    def __getitem__(self, idx):
        return SamplesAccumulator(self.values[idx], self.alphas[idx])

class Percentiles(object):
    def __init__(self):
        self.percentiles = None

    def add(self, xs):
        self.percentiles = xs.concat_to(self.percentiles).sortval()

    def at(self, percentile):
        assert 0 <= percentile <= 1
        n = len(self)
        q = int(percentile * n) - 1
        if q < 0:
            q = 0
        return self.percentiles[q]

    def __len__(self):
        return len(self.percentiles) if self.percentiles else 0

class QuasiRejectionSamplerFixedBeta(BaseSampler):

    def __init__(self, ebm, proposal, beta, batch_size, silent=False):
        super(QuasiRejectionSamplerFixedBeta, self).__init__(ebm, proposal)
        self.batch_size = batch_size
        self.beta = beta
        self.accepted = 0
        self.total = 0
        self.silent = silent

    def sample(self):
        samples = SamplesAccumulator()
        alpha_pcts = Percentiles()
        if not self.silent:
            prog = tqdm.tqdm(desc="QRS", total=self.batch_size if self.batch_size else 1)
        while self.batch_size is None or len(samples) < self.batch_size:
            x, beta_x = self._sample_and_assess_importance(self.batch_size - len(samples) if self.batch_size else None)
            u = beta_x.new_uniform(0, 1)
            alpha_x = beta_x / u
            alpha_pcts.add(alpha_x)
            x = x.filter(alpha_x > self.beta)
            alpha_x = alpha_x.filter(alpha_x > self.beta)
            self.accepted += len(x)
            if not self.silent:
                prog.update(len(x))
                prog.set_postfix(beta=self.beta, ar=self.get_acceptance_rate())
            samples.add(x, alpha_x)
            if self.batch_size is None:
                break
        if not self.silent:
            prog.close()
        return samples.values

    def _sample_and_assess_importance(self, max_qty):
        assert max_qty is None or max_qty > 0
        x = self.proposal.sample()
        if max_qty is not None and len(x) > max_qty:
            x = x[:max_qty]
        self.total += len(x)
        try:
            q_x = self.proposal.log_score(x)
            P_x = self.ebm.log_score(x)
            beta_x = (P_x - q_x).exp()
        except AttributeError:
            q_x = self.proposal.score(x)
            P_x = self.ebm.score(x)
            beta_x = P_x / q_x
        return x, beta_x

    def get_acceptance_rate(self):
        return self.accepted / self.total

    def get_tvd_bound(self, delta=0.05):
        return np.sqrt(self.beta ** 2 / self.total / 2 * (-np.log(delta)))

    def get_meta(self):
        return {
                'batch_size': self.batch_size,
                'beta': self.beta,
                'tvd_bound:d=0.05': self.get_tvd_bound(delta=0.05),
                'accepted': self.accepted,
                'total': self.total}
