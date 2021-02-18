# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import numpy as np
from . import BaseSampler, AccumulatorSampler
from discontrol.misc import ListArray

class IMHResetSampler(BaseSampler):
    
    def __init__(self, ebm, proposal, burn_in, batch_size=None):
        """
        Stateless independent metropolis-hastings sampler that resets the chain for every sample.
        """
        super(IMHResetSampler, self).__init__(ebm, proposal)
        self.burn_in = burn_in
        self.batch_size = proposal.batch_size if batch_size is None else batch_size

    def sample(self):
        samples = []
        chain = []
        last = None
        logp_last = None
        logq_last = None

        while len(samples) < self.batch_size:
            x = self.proposal.sample()        
            log_q = self.proposal.log_score(x)
            log_p = self.ebm.log_score(x)
            u = log_p.new_uniform(0, 1)

            for i, xi in enumerate(x):

                # First proposal is always accepted.
                if last is None:
                    chain.append(xi)
                    logp_last = log_p[i]
                    logq_last = log_q[i]
                    last = xi
                    continue
                
                alpha = min(1., np.exp((log_p[i] - logp_last) + (logq_last - log_q[i])))
                if u[i] <= alpha:
                    
                    # Accept the new state.
                    chain.append(xi)
                    logp_last = log_p[i]
                    logq_last = log_q[i]
                    last = xi
                else:
                     
                    # Reject the new state, keep the old state.
                    chain.append(last)

                # Add sample and reset chain after every burn_in period.
                if len(chain) > self.burn_in:
                    samples.append(chain[-1])
                    chain = []
                    last = None
                    logp_last = None
                    logq_last = None
                if len(samples) >= self.batch_size: break

        return  ListArray(samples)

    def get_meta(self):
        return {
            "batch_size": self.batch_size,
            "burn_in": self.burn_in,
        }
