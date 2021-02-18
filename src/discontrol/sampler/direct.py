# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from . import BaseSampler, AccumulatorSampler

class DirectSampler(BaseSampler):
    """Ignores the ebm and samples directly from the proposal"""

    def __init__(self, ebm, proposal, batch_size):
        accum = AccumulatorSampler(proposal, batch_size)
        super(DirectSampler, self).__init__(ebm, accum)

    def sample(self):
        return self.proposal.sample()

    def get_acceptance_rate(self):
        return 1

    def get_meta(self):
        meta = super(DirectSampler, self).get_meta()
        try:
            meta.update(self.proposal.get_meta())
        except:
            pass
        return meta

