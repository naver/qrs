# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

class BaseScorer(object):

    def __mul__(self, ot):
        # FIXME: hacky
        from discontrol.scorer.product import ProductScorer
        return ProductScorer(self, ot)

    def score_batched(self, samples, batch_size):
        all_scores = []
        for i in range(len(samples)//batch_size + 1):
            batch_samples = samples[i*batch_size:(i+1)*batch_size]
            if batch_samples:
                all_scores.append(self.score(batch_samples))
        return type(all_scores[0]).cat(all_scores)

    # FIXME: factorize
    def log_score_batched(self, samples, batch_size):
        all_scores = []
        for i in range(len(samples)//batch_size + 1):
            batch_samples = samples[i*batch_size:(i+1)*batch_size]
            if batch_samples:
                all_scores.append(self.log_score(batch_samples))
        return type(all_scores[0]).cat(all_scores)

    def log_score(self, samples):
        return self.score(samples).log()

    def to(self, device):
        return self
