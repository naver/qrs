# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import numpy as np
from . import BaseSampler, AccumulatorSampler
from discontrol.misc import list_array

class RWMHSampler(BaseSampler):
    
    def __init__(self, ebm, proposal, q_local, batch_size=None, keep_every=1, burn_in=0):
        super(RWMHSampler, self).__init__(ebm, q_local)
        self.keep_every = keep_every
        self.burn_in = burn_in
        self.burnt_in = False
        self.batch_size = proposal.batch_size if batch_size is None else batch_size
        self.accepted = 0
        self.total = 0
        self.q_local = q_local
        q_seed = proposal

        q_seed.batch_size = 1
        self.last = q_seed.sample()[0] 
        self.logp_last = ebm.log_score(list_array([self.last]))[0]

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
            x, logq, logq_inv = self.q_local.sample_and_score(self.last)
            logp = self.ebm.log_score(list_array([x]))[0]
            u = np.random.uniform()
            
            alpha = min(1., np.exp((logp - self.logp_last) + (logq_inv - logq))) 
            alphas.append(alpha)

            if u <= alpha:
                chain.append(x)
                self.logp_last = logp
                self.last = x
                accepted.append(1)
            else:
                chain.append(self.last)
                accepted.append(0)
    
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
            return  list_array(samples)

    def get_acceptance_rate(self):
        return (1 / self.keep_every) * (1 - self.burn_in / self.total)

