# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import streamlit as st
from discontrol.distribution.poisson import PoissonDistribution
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from stqdm import stqdm
pd.set_option('display.float_format', '{:.4g}'.format)

ci='sd'

def main():
    pass
    lam_P = st.number_input(label='λP', min_value=1, max_value=20, value=11)
    lam_q = st.number_input(label='λq', min_value=1, max_value=20, value=10)
    p = PoissonDistribution(lam=lam_P)
    q = PoissonDistribution(lam=lam_q)
    beta = st.number_input('beta',min_value=0., max_value=100.,step=.1, value=2.)

    Z = 1
    k = int((lam_P - lam_q + np.log(beta)) / (np.log(lam_P) - np.log(lam_q)))
    Z_beta = stats.poisson.cdf(k, lam_P) + beta * (1 - stats.poisson.cdf(k, lam_q))
    st.write('Z_beta', Z_beta)

    x = np.arange(10**4)
    logP_x = stats.poisson.logpmf(x, lam_P)
    logq_x = stats.poisson.logpmf(x, lam_q)
    P_x = np.exp(logP_x)
    q_x = np.exp(logq_x)
    logP_beta_x = np.minimum(np.log(beta) + logq_x, logP_x)
    logp_beta_x = logP_beta_x - np.log(Z_beta)
    logp_x = logP_x - np.log(Z)
    p_beta_x = np.exp(logp_beta_x)
    p_x = np.exp(logp_x)


    true_tvd = np.abs(p_beta_x - p_x).sum() / 2
    true_kl = (p_x * np.nan_to_num(logp_x - logp_beta_x)).sum()
    st.write('True TVD', true_tvd)
    st.write('True KL', true_kl)

    repetitions = 1000
    est = []
    with Pool(8) as p:
        for n_samples in stqdm([100, 1000, 10_000, 100_000, 1_000_000]):
            for i in stqdm(range(repetitions)):
                tvd_est, kl_est = estimate_divergence(n_samples, beta, lam_P, lam_q)
                est.append({'n_samples': n_samples, 'i': i, 'tvd_est': tvd_est, 'kl_est': kl_est})

    df = pd.DataFrame(est)
    df['tvd_bias'] = df['tvd_est'] - true_tvd
    df['kl_bias'] = df['kl_est'] - true_kl
    
    st.write('mean')
    st.write(df.groupby('n_samples').mean().style.format('{:.4g}'))
    st.write('std')
    st.write(df.groupby('n_samples').std().style.format('{:.4g}'))

    fig = plt.figure(figsize=(7,2))
    sns.lineplot(data=df, x='n_samples', y='tvd_bias', ci=ci)
    plt.xscale('log')
    plt.axhline(y = 0, linestyle = '--', color='C1')
    plt.ylabel(r"Est. TVD $-$ True TVD")
    plt.xlabel("number of samples")
    st.write(fig)
    fig.savefig(f'figures/TVD-bias-{beta}.png', dpi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(7,2))
    sns.lineplot(data=df, x='n_samples', y='kl_bias', ci=ci)
    plt.xscale('log')
    plt.axhline(y = 0, linestyle = '--', color='C1')
    plt.ylabel(r"Est. KL $-$ True KL")
    plt.xlabel("number of samples")
    fig.savefig(f'figures/KL-bias-{beta}.png', dpi=300, bbox_inches='tight')
    st.write(fig)


def estimate_divergence(n_samples, beta, lam_P, lam_q):
    x = stats.poisson.rvs(lam_q, size=n_samples)
    q_x = stats.poisson.pmf(x, lam_q)
    P_x = stats.poisson.pmf(x, lam_P)
    P_beta_x = np.minimum(beta * q_x , P_x)
    Z_beta_est = np.mean(P_beta_x / q_x)
    Z_est = np.mean(P_x / q_x)
    p_beta_x = P_beta_x / Z_beta_est
    p_x = P_x / Z_est
    tvd_est = np.abs(p_beta_x/q_x - p_x/q_x).mean() / 2
    kl_est = (p_x / q_x * (np.log(p_x) - np.log(p_beta_x))).mean()
    return tvd_est, kl_est

main()
