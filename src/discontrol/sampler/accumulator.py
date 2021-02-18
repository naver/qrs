# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import tqdm

class AccumulatorSampler(object):
    """
    Accumulates several batches until the given size is reached
    """

    def __init__(self, sampler, batch_size, silent=True):
        self.sampler = sampler
        self.batch_size = batch_size
        self.silent = True

    def sample(self):
        with tqdm.tqdm(total=self.batch_size, desc=f'Sampling from {type(self.sampler).__name__}') as prog:
            remaining = self.batch_size
            sampled = None
            while remaining > 0:
                new_sample = self.sampler.sample()
                if len(new_sample) > remaining:
                    new_sample = new_sample[:remaining]
                sampled = new_sample.concat_to(sampled)
                remaining -= len(new_sample)
                prog.update(len(new_sample))
            return sampled

    def get_meta(self):
        return self.sampler.get_meta()

    def get_acceptance_rate(self):
        return self.sampler.get_acceptance_rate()

