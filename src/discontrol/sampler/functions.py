# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from .accumulator import AccumulatorSampler

def sample_until(sampler, N, silent=True):
    accum = AccumulatorSampler(sampler, batch_size=N, silent=silent)
    return accum.sample()

