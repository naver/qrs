# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from scipy import stats
from discontrol.misc import NumpyArray
from discontrol.scorer import BaseScorer

class PoissonDistribution(BaseScorer):
    
    def __init__(self, lam, batch_size=1):
        self.lam = lam
        self.batch_size = batch_size

    def sample(self):
        return NumpyArray(stats.poisson.rvs(self.lam, size=self.batch_size))

    def score(self, x):
        return self.normalized_score(x)

    def log_score(self, x):
        return NumpyArray(stats.poisson.logpmf(x, self.lam))

    def normalized_score(self, x):
        return NumpyArray(stats.poisson.pmf(x, self.lam))


class TruncatedPoissonDistribution(BaseScorer):
    
    def __init__(self, lam, upper, batch_size=1):
        self.lam = lam
        self.upper = upper
        self.batch_size = batch_size

    def sample(self):
        raise NotImplementedException()

    def score(self, x):
        return self.normalized_score(x)

    def normalized_score(self, x):
        assert x < self.upper
        Z = stats.poisson.cdf(self.upper - 1, self.lam)
        return NumpyArray(stats.poisson.pmf(x, self.lam) / Z)


