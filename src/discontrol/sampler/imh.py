# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import numpy as np
from . import BaseSampler, AccumulatorSampler
from discontrol.misc import ListArray

class IMHSampler(BaseSampler):
    
    def __init__(self, ebm, proposal, batch_size=None, keep_every=1, burn_in=0):
        """
        Stateful independent metropolis-hastings sampler.
        """
        super(IMHSampler, self).__init__(ebm, proposal)
        self.keep_every = keep_every
        self.burn_in = burn_in
        self.burnt_in = False
        self.batch_size = proposal.batch_size if batch_size is None else batch_size
        self.accepted = 0
        self.total = 0

        self.last = None
        self.logp_last = None
        self.logq_last = None

    def sample(self, return_diagnostics=False):
        chain = []
        alphas = []
        accepted = []

        # For the first call to sample(.) we need to run for the burn-in time.
        if not self.burnt_in:
            req_length = self.burn_in + self.batch_size * self.keep_every
        else:
            req_length = self.batch_size * self.keep_every

        while len(chain) < req_length:
            x = self.proposal.sample()        
            log_q = self.proposal.log_score(x)
            log_p = self.ebm.log_score(x)
            u = log_p.new_uniform(0, 1)

            for i, xi in enumerate(x):

                # First proposal is always accepted.
                if self.last is None:
                    chain.append(xi)
                    self.logp_last = log_p[i]
                    self.logq_last = log_q[i]
                    self.last = xi
                    accepted.append(1)
                    continue
                
                alpha = min(1., np.exp((log_p[i] - self.logp_last) + (self.logq_last - log_q[i])))
                alphas.append(alpha)

                if u[i] <= alpha:
                    
                    # Accept the new state.
                    chain.append(xi)
                    self.logp_last = log_p[i]
                    self.logq_last = log_q[i]
                    self.last = xi
                    accepted.append(1)
                else:
                     
                    # Reject the new state, keep the old state.
                    chain.append(self.last)
                    accepted.append(0)

                if len(chain) >= req_length: break

        # Discard the burn-in samples.
        if not self.burnt_in:
            samples = chain[self.burn_in::self.keep_every]
            self.burnt_in = True
        else:
            samples = chain[::self.keep_every]

        self.accepted += sum(accepted)
        self.total += len(accepted)

        if return_diagnostics:
            diagnostics = {
                "chain": chain,
                "alphas": alphas,
                "accepted": accepted
            }
            return samples, diagnostics
        else:
            return  ListArray(samples)

    def get_acceptance_rate(self):
        return (1 / self.keep_every) * (1 - self.burn_in / self.total)

    def get_meta(self):
        return {
            "batch_size": self.batch_size,
            "keep_every": self.keep_every,
            "burn_in": self.burn_in,
            "accepted": self.accepted,
            "total": self.total,
            "raw_acceptance_rate": self.accepted / self.total
        }
