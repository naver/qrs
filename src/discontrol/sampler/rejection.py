# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from . import BaseSampler, AccumulatorSampler
import numpy as np

class RejectionSampler(BaseSampler):
    def __init__(self, ebm, proposal, beta, truncate=False, batch_size=None):
        super(RejectionSampler, self).__init__(ebm, proposal)
        aux_sampler = RejectionSamplerAux(ebm, proposal, beta, truncate)
        if batch_size is not None:
            self.sampler = AccumulatorSampler(aux_sampler, batch_size)
        else:
            self.sampler = aux_sampler

    def sample(self):
        return self.sampler.sample()

    def get_acceptance_rate(self):
        return self.sampler.get_acceptance_rate()
            

class RejectionSamplerAux(BaseSampler):

    def __init__(self, ebm, proposal, beta, truncate):
        super(RejectionSamplerAux, self).__init__(ebm, proposal)
        self.beta = beta
        self.truncate = truncate
        self.accepted = 0
        self.total = 0

    def sample(self):
        x = self.proposal.sample()
        log_q = self.proposal.log_score(x)
        log_P = self.ebm.log_score(x)
        r = (log_P - log_q).exp() / self.beta
        if self.truncate:
            r = np.minimum(r, 1)
        if (r > 1).any():
            raise RuntimeError(f"P/(q beta) (={r.max()}) > 1")
        u = r.new_uniform(0, 1)
        filtered_x = x.filter(u < r)
        self.accepted += len(filtered_x)
        self.total += len(x)
        return filtered_x

    def get_acceptance_rate(self):
        return self.accepted / self.total

