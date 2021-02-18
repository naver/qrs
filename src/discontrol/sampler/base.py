# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from abc import abstractmethod

class BaseSampler(object):
    def __init__(self, ebm, proposal):
        self.proposal = proposal
        self.ebm = ebm

    @abstractmethod
    def sample(self):
        pass

    def get_meta(self):
        return {}
