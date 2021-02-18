# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import os
import sys
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from discontrol.sampler import QuasiRejectionSampler, IMHSampler
from discontrol.distribution.poisson import PoissonDistribution

# Plot settings
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
         '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

def main():
    st.title("Independent Metropolis-Hastings (Poissons)")

    sample_size = st.slider(label="Number of samples", min_value=100, max_value=5000, value=2500)
    rate_p = st.slider(label="λ1", min_value=1, max_value=20, value=11)
    rate_q = st.slider(label="λ2", min_value=1, max_value=20, value=20)
    burn_in = st.slider(label="burn-in", min_value=0, max_value=5000, value=2500)
    keep_every = st.slider(label="keep every kth sample", min_value=1, max_value=100, value=1)
    p = PoissonDistribution(lam=rate_p, batch_size=sample_size)
    q = PoissonDistribution(lam=rate_q, batch_size=sample_size)

    imh = IMHSampler(p, q, batch_size=sample_size, keep_every=keep_every, burn_in=burn_in)
    
    s_exact = p.sample()
    s_prop = q.sample()
    # s_qrs = qrs.sample()
    s_imh, imh_diag = imh.sample(return_diagnostics=True)

    st.write(f"Got {len(s_imh)} samples from IMH.")
    st.write(f"IMH raw acceptance rate={np.mean(imh_diag['accepted']):.2f}")

    # Plot p and q.
    fig = plot_histogram([s_exact, s_prop], ["exact", "proposal"],
                         title=f"p=Poisson({rate_p}) q=Poisson({rate_q})")
    st.pyplot(fig)

    # Exact sampling vs IMH.
    fig = plot_histogram([s_exact, s_imh], ["exact", "IMH"], title=f"Poisson({rate_p})")
    st.pyplot(fig)

    # IMH chain
    fig = plot_imh_chain(imh_diag["chain"])
    st.pyplot(fig)

    # Plot TVD for different sampling techniques.
    fig = plot_tvd([s_exact, s_prop, s_imh], ["exact", "proposal", "IMH"], rate_p, title="TVD")
    #fig = plot_tvd([s_exact, s_prop, s_qrs, s_imh], ["exact", "proposal", "QRS", "IMH"], rate_p, title="TVD")
    st.pyplot(fig)
    
    # Plot moments for different sampling techniques.
    fig = plot_moments([s_exact, s_prop, s_imh], ["exact", "proposal", "IMH"], 
                       rate_p, title=f"Poisson({rate_p}) moments")
    # fig = plot_moments([s_exact, s_prop, s_qrs, s_imh], ["exact", "proposal", "QRS", "IMH"], 
    #                     rate_p, title=f"Poisson({rate_p}) moments")
    st.pyplot(fig)

    # Plot TVD vs burn-in.
    fig = plot_tvd_burn_in(p, q, sample_size, rate_p)
    st.pyplot(fig)

    # Plot TVD vs thinning.
    fig = plot_tvd_thinning(p, q, sample_size, rate_p)
    st.pyplot(fig)

    # Plot TVD vs thinning.
    # fig = plot_tvd_thinning(p, q, sample_size, rate_p, burn_in=10000)
    # st.pyplot(fig)

def plot_imh_chain(chain):
    fig, ax = plt.subplots()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sample Value")
    ax.set_title("IMH chain")
    ax.plot(np.arange(len(chain)), chain)
    return fig

def plot_tvd_burn_in(p, q, sample_size, true_rate, keep_every=1):
    burn_ins = [0, 1000, 10000, 100000]
    tvds = []
    nreps = 10
    for burn_in in burn_ins:
        tvd = 0
        for reps in range(nreps):
            imh = IMHSampler(p, q, batch_size=sample_size, keep_every=keep_every, burn_in=burn_in)
            s_imh = imh.sample()
            tvd += compute_tvd_poisson(s_imh, true_rate)
        tvds.append(tvd / nreps)
    fig, ax = plt.subplots()
    ax.scatter(burn_ins, tvds, marker='x')
    ax.set_xlabel("Burn-in duration")
    ax.set_ylabel("average TVD")
    ax.set_xscale('symlog')
    ax.set_title(f"TVD as a function of burn-in length averaged over {nreps} runs (keep-every={keep_every}).")
    return fig

def plot_tvd_thinning(p, q, sample_size, true_rate, burn_in=0):
    ks = [1, 5, 10, 20, 30, 40, 50]
    tvds = []
    nreps = 10
    for keep_every in ks:
        tvd = 0
        for reps in range(nreps):
            imh = IMHSampler(p, q, batch_size=sample_size, keep_every=keep_every, burn_in=burn_in)
            s_imh = imh.sample()
            tvd += compute_tvd_poisson(s_imh, true_rate)
        tvds.append(tvd / nreps)
    fig, ax = plt.subplots()
    ax.scatter(ks, tvds, marker='x')
    ax.set_xlabel("Keep every")
    ax.set_ylabel("average TVD")
    ax.set_title(f"TVD as a function of thinning averaged over {nreps} runs (burn-in={burn_in}).")
    return fig

def compute_tvd_poisson(samples, rate):
    outcomes, counts = np.unique(samples, return_counts=True)
    empirical_dist = dict(zip(outcomes, counts / np.sum(counts)))
    
    tvd = 0
    for x in range(0, np.max(outcomes)+1):
        if x in empirical_dist: q = empirical_dist[x]
        else: q = 0.
        p = scipy.stats.poisson.pmf(x, rate)
        tvd += np.abs(p-q)

    remaining = 1.0 - scipy.stats.poisson.cdf(np.max(outcomes), rate)
    tvd += remaining

    return tvd / 2.

def plot_tvd(sets, labels, true_rate, title=None):
    fig, ax = plt.subplots()
    ax.set_title(title)
    tvds = []
    
    for samples, label in zip(sets, labels):
        tvds.append(compute_tvd_poisson(samples, true_rate))

    df = pd.DataFrame({
        "TVD": tvds,
        "labels": labels,
        "sampling method": [""] * len(labels)
    })

    sns.barplot(x="sampling method", y="TVD", hue="labels", data=df, ax=ax)
    ax.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)

    return fig

def plot_histogram(sets, labels, title=None):
    fig, ax = plt.subplots()
    ax.set_title(title)
    i = 0
    for samples, label in zip(sets, labels):
        sns.histplot(data=samples, ax=ax, label=label, binwidth=1, color=colors[i % len(colors)], bins=25)
        i += 1
    ax.legend()
    return fig

def plot_moments(sets, labels, true_rate, title=None):
    fig, axes = plt.subplots(2, 1, figsize=(5, 10))
    axes[0].set_title(title)
    axes[1].set_title(title)
    means = []
    variances = []
    skews = []
    kurts = []
    
    for samples, label in zip(sets, labels):
        means.append(np.average(samples))
        variances.append(np.var(samples))
        skews.append(scipy.stats.skew(samples))
        kurts.append(scipy.stats.kurtosis(samples))

    means.append(scipy.stats.poisson.stats(true_rate, moments='m').item())
    variances.append(scipy.stats.poisson.stats(true_rate, moments='v').item())
    skews.append(scipy.stats.poisson.stats(true_rate, moments='s').item())
    kurts.append(scipy.stats.poisson.stats(true_rate, moments='k').item())
    labels.append("truth")

    moments1 = means + variances
    df1 = pd.DataFrame({
        "moments": moments1,
        "labels": labels * 2,
        "x": ["mean"] * len(labels) + ["variance"] * len(labels)
    })

    moments2 = skews + kurts
    df2 = pd.DataFrame({
        "moments": moments2,
        "labels": labels * 2,
        "x": ["skewness"] * len(labels) + ["kurtosis"] * len(labels)
    })

    sns.barplot(x="x", y="moments", hue="labels", data=df1, ax=axes[0])
    axes[0].legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
    axes[0].set_xlabel("")
    sns.barplot(x="x", y="moments", hue="labels", data=df2, ax=axes[1])
    axes[1].legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
    axes[1].set_xlabel("")

    return fig

if __name__ == "__main__":
    main()
