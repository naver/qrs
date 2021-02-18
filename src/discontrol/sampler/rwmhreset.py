# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import numpy as np
from . import BaseSampler, AccumulatorSampler
from discontrol.misc import list_array

class RWMHResetSampler(BaseSampler):

    def __init__(self, ebm, proposal, reset_every, q_local, batch_size=None):
        """ 
        Stateless RWMH sampler.
        """
        super(RWMHResetSampler, self).__init__(ebm, q_local)
        self.reset_every = reset_every
        self.q_local = q_local
        self.q_seed = proposal   
        self.q_seed.batch_size = 1
        self.batch_size = batch_size

    def sample(self):
        samples = []
        last = self.q_seed.sample()[0]
        logp_last = self.ebm.log_score(list_array([last]))[0]
        chain = [last]
        
        while len(samples) < self.batch_size:
            x, logq, logq_inv = self.q_local.sample_and_score(last)
            logp = self.ebm.log_score(list_array([x]))[0]
            u = np.random.uniform()
            alpha = min(1., np.exp((logp - logp_last) + (logq_inv - logq))) 

            if u <= alpha:
                chain.append(x)
                logp_last = logp
                last = x
            else:
                chain.append(last)

            if len(chain) > self.reset_every:
                samples.append(chain[-1])
                print(samples[-1])
                chain = []
                last  = self.q_seed.sample()[0]
                logp_last = self.ebm.log_score(list_array([last]))[0]
    
        return  list_array(samples)

    def get_acceptance_rate(self):
        return (1 / self.reset_every)
