# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from scipy import stats
import numpy as np

def g_test(normalized_distr, obs_freqs):
    Z = sum(obs_freqs.values())
    X = sorted(list(obs_freqs.keys()))
    exp_freqs = [normalized_distr.normalized_score(x) * Z for x in X]
    v_obs_freqs = [obs_freqs[x] for x in X]
    return stats.power_divergence(v_obs_freqs , exp_freqs, lambda_="log-likelihood")

def kl_exp_obs(normalized_distr, obs_freqs):
    Z = sum(obs_freqs.values())
    X = sorted(list(obs_freqs.keys()))
    exp_prob = [normalized_distr.normalized_score(x) for x in X]
    exp_prob.append(1. - sum(exp_prob))
    obs_prob = [obs_freqs[x]/Z for x in X]
    obs_prob.append(0)
    return stats.entropy(obs_prob, exp_prob)

def tvd_exp_obs(normalized_distr, obs_freqs):
    X = sorted(list(obs_freqs.keys()))
    exp_prob = np.array([normalized_distr.normalized_score(x) for x in X])
    Z = sum(obs_freqs.values())
    obs_prob = np.array([obs_freqs[x]/Z for x in X])
    unobserved_mass = 1. - sum(exp_prob)
    return (np.sum(np.abs(exp_prob - obs_prob)) + unobserved_mass) / 2
