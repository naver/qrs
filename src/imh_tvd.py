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
import torch
import multiprocessing

import sacrebleu
import nltk
from nltk.translate.bleu_score import SmoothingFunction

from pathlib import Path

from discontrol.sampler import QuasiRejectionSampler, IMHSampler
from discontrol.distribution.samplerepo import SampleRepository
from discontrol.scorer.common import broadcast
from discontrol.scorer import SingleWordFeature, GenderFeature, MultiWordFeature

from qrs_tvd import  get_tvd_estimates, add_bounds, project_into_curve_x

# Plot settings
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
         '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

@st.cache(allow_output_mutation=True)
def load_samples(samples_file, ebm_scores_file, proposal_scores_file):
    p = SampleRepository(samples_file, ebm_scores_file)
    q = SampleRepository(samples_file, proposal_scores_file)
    return p, q

def main():
    #st.set_page_config(layout="wide")

    data_folder = Path("<output-dir>/female_science/")
    samples_file = data_folder / "direct-female_science.pkl"
    ebm_scores_file = data_folder / "direct-female_science.ebm.pkl"
    proposal_scores_file = data_folder / "direct-female_science.proposal.pkl"

    st.title("Independent Metropolis-Hastings (Sequences)")

    sample_size = st.slider(label="sample size", min_value=100, max_value=500_000, value=50_000)
    burn_in = st.slider(label="burn-in", min_value=0, max_value=5000, value=0)
    keep_every = st.slider(label="keep every kth sample", min_value=1, max_value=100, value=1)

    # Cached data loading
    st.write("Loading samples...")
    p, q = load_samples(samples_file, ebm_scores_file, proposal_scores_file)
    q.batch_size = sample_size
    st.write(f"Loaded {p.size:,} samples from disk.")

    # Distribution of TVD lower-bound estimates.
    _ = """
    nreps = 100
    nbins, bin_func = (50, broadcast(lambda x: min(50, len(x.split(' ')))))
    scorer = SingleWordFeature("amazing")
    tvds_naive = compute_naive_tvd_bounds(nreps, nbins, bin_func, scorer, p, q)
    tvds_imh = compute_imh_tvd_bounds(nreps, nbins, bin_func, scorer, p, q, sample_size, keep_every, burn_in)
    fig, ax = plt.subplots()
    ax.set_xlabel("TVD lower-bound")
    ax.set_title(f"TVD lower-bounds over {nreps} repetitions")
    sns.histplot(data=tvds_naive, ax=ax, label="naive", color=colors[0], binwidth=0.01)
    sns.histplot(data=tvds_imh, ax=ax, label="IMH", color=colors[1], binwidth=0.01)
    ax.legend()
    st.pyplot(fig)
    """
    
    # Naive filter samples
    # scorer = SingleWordFeature("amazing")
    scorer_female = GenderFeature("female")
    scorer_science = MultiWordFeature("resources/wikibio-wordlists/science.txt")
    scorer = scorer_science
    s_q = q.sample()
    scored_sq = scorer.score(s_q)
    s_naive = [x for (x, b) in zip(s_q, scored_sq) if b]
    ar_naive = len(s_naive) / len(s_q)
    st.write(f"Naive filter acceptance rate = {ar_naive:.2f}")

    # Collect IMH samples 
    imh = IMHSampler(p, q, batch_size=sample_size, keep_every=keep_every, burn_in=burn_in)
    s_imh, imh_diag = imh.sample(return_diagnostics=True)
    st.write(f"Got {len(s_imh):,} samples from IMH.")
    ar_imh = len(s_imh) / imh.total
    st.write(f"IMH raw acceptance rate={np.mean(imh_diag['accepted']):.2f}")
    st.write(f"IMH #samples/#proposals={ar_imh:.2f}")

    # QRS samples
    qrs_min_acceptance = 0.1
    qrs = QuasiRejectionSampler(p, q, sample_size, min_acceptance_rate=qrs_min_acceptance)
    s_qrs = qrs.sample() 
    st.write(f"QRS min acceptance rate = {qrs_min_acceptance:.2f} - QRS acceptance rate = {qrs.get_acceptance_rate():.2f}")

    fig = plot_constraint_satisfaction([s_q, s_naive, s_imh, s_qrs], ["proposal", "naive", "IMH", "QRS"], scorer_science, constraint_name="science", target=100)
    st.pyplot(fig)

    fig = plot_constraint_satisfaction([s_q, s_naive, s_imh, s_qrs], ["proposal", "naive", "IMH", "QRS"], scorer_female, constraint_name="female", target=50)
    st.pyplot(fig)
    
    _ = """ expensive
    fig = plot_self_bleu([s_q, s_naive, s_imh, s_qrs], ["proposal", "naive", "IMH", "QRS"])
    st.pyplot(fig)
    """

    # Plot unique fraction
    uniq_q = len(np.unique(s_q)) / len(s_q)
    uniq_naive = len(np.unique(s_naive)) / len(s_naive)
    uniq_imh = len(np.unique(s_imh)) / len(s_imh)
    uniq_qrs = len(np.unique(s_qrs)) / len(s_qrs)
    df = pd.DataFrame({
        "% unique samples": [uniq_q, uniq_naive, uniq_imh, uniq_qrs],
        "labels": ["proposal", "naive", "IMH", "QRS"],
        "sampling method": [""] * 4
    })
    fig, ax = plt.subplots()
    ax.set_title("% unique samples for different sampling methods")
    sns.barplot(x="sampling method", y="% unique samples", hue="labels", data=df, ax=ax)
    ax.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
    st.pyplot(fig)


    # Plot binning by length
    bin_by_length = broadcast(lambda x: min(40, len(x.split(' '))))
    binned_imh = bin_by_length(s_imh)
    binned_proposal = bin_by_length(s_q)
    fig, ax = plt.subplots()
    ax.set_xlabel("length")
    ax.set_title(f"Binning by length")
    sns.histplot(data=binned_proposal, ax=ax, label="proposal", color=colors[0], binwidth=1)
    sns.histplot(data=binned_imh, ax=ax, label="IMH", color=colors[1], binwidth=1)
    ax.legend()
    st.pyplot(fig)

    # TVD lower-bound
    bin_funcs = {
            # '(binned by word count "amazing")': (nbins, broadcast(lambda x: min(x.count('amazing'), nbins+1)-1)),
            '(binned by length)': (50, bin_by_length)
    }
    samplers = ["naive", "IMH"]
    acceptance_rates = [ar_naive, ar_imh]
    samples = [s_naive, s_imh]
    logp_sq = p.log_score(s_q)
    logq_sq = q.log_score(s_q)
    binned_tvds = {}
    for sampler_name, ar, s in zip(samplers, acceptance_rates, samples):
        for bin_method_name, (nbins, bin_func) in bin_funcs.items():
            binned_tvds[f"{sampler_name}  {bin_method_name}"] = (ar, compute_binned_tvd(s_q, logp_sq, logq_sq, s, bin_func, nbins))
    st.write(binned_tvds)

    # Compare with QRS and the naive filter.
    log_unnorm_beta_x = (logp_sq - logq_sq).exp()
    norm_logp_sq = logp_sq - np.log(log_unnorm_beta_x.mean())
    beta_x = (norm_logp_sq - logq_sq).exp()
    beta_tick_size = (beta_x.max() - beta_x.min()) / 1000.
    beta_ticks = np.arange(beta_x.min(), beta_x.max(), beta_tick_size)
    delta = 0.05
    df_data = get_tvd_estimates(beta_x, beta_ticks, scored_sq)
    df_data = add_bounds(df_data, beta_ticks, len(logp_sq), delta)
    fig, ax = plt.subplots()
    sns.lineplot(data=df_data, x="ar", y="tvd", ax=ax, label="QRS bound")
    sns.lineplot(data=df_data, x="ar", y='tvd_with_bound', ax=ax,  label='QRS Hoeffding bound')
    
    for bin_method_name, val in binned_tvds.items():
        ar, tvd = val
        ax.plot([ar], [tvd], marker='x', linestyle='None', markersize=10, label=bin_method_name)
        # follow https://stackoverflow.com/questions/65541863/pyplot-adding-point-projections-on-axis
        plt.annotate(f'{tvd:.1e}', xy=(ar, tvd), xytext=(0, tvd), 
                     textcoords=plt.gca().get_yaxis_transform(),
                     arrowprops={'arrowstyle': '-', 'ls':'--', 'color': 'grey'}, 
                     size=8,
                     va='center', ha='right') 
    ax.set_ylabel('TVD')
    ax.set(yscale='log')
    ax.set(xscale='log')
    ax.set_xlabel("acceptance rate")
    ax.invert_xaxis()
    ax.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
    st.pyplot(fig)

    # Example samples
    st.write("## Example Samples")
    st.write(s_imh[:10])

def plot_self_bleu(sets, labels):

    self_bleus = []
    for samples in sets:
        self_bleus.append(self_bleu(samples))
    df = pd.DataFrame({
        "self-BLEU": self_bleus,
        "labels": labels,
        "sampling method": [""] * len(labels)
    })

    fig, ax = plt.subplots()
    ax.set_title("self-BLEU for different samplers")
    sns.barplot(x="sampling method", y="self-BLEU", hue="labels", data=df, ax=ax)
    ax.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)

    return fig

def plot_constraint_satisfaction(sets, labels, scorer, constraint_name, target):
    fig, ax = plt.subplots()
    ax.set_title(f"Constraint Satisfaction of constraint={target}% {constraint_name} for different samplers")
    cs = []
    for samples in sets:
        sat = scorer.score(samples)
        cs.append(np.mean(sat)) 

    df = pd.DataFrame({
        "CS": cs,
        "labels": labels,
        "sampling method": [""] * len(labels)
    })

    sns.barplot(x="sampling method", y="CS", hue="labels", data=df, ax=ax)
    ax.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
    ax.axhline(y=target/100., color="black", linestyle="dashed")

    return fig


def compute_binned_tvd(x_q, log_P_x, log_q_x, s, bin_func, nbins):
    """
    :param x_q: samples from q(x)
    :param log_P_x: unnormalized log scores of the samples from the EBM log P(x)
    :param log_p_q: log probabilities of the samples from the proposal log q(x)
    :param s: samples from an arbitrary sampler
    :param bin_func: function that assigns bins to a sample
    :param nbins: number of bins used
    """

    # P_proj_ebm - SNIS estimate
    binned_x_q = np.array(bin_func(x_q))
    p_proj_ebm = np.zeros(nbins)
    unnormalized_iw = (log_P_x - log_q_x).exp()
    sn_log_iw = unnormalized_iw.log() - np.log(unnormalized_iw.sum())
    # st.write(f"{(sn_log_iw).exp()[:10]} {(sn_log_iw).exp().max()} {(sn_log_iw.exp().sum())}")
    for b in range(nbins):
        p_proj_ebm[b] = ((sn_log_iw).exp() * (binned_x_q == b)).sum()
    p_proj_ebm /= p_proj_ebm.sum()

    # P_proj_sampler
    binned_s = np.array(bin_func(s))
    bin_counts_s = np.zeros(nbins)
    for b in range(nbins):
        bin_counts_s[b] += (binned_s == b).sum()
    p_proj_sampler = bin_counts_s / bin_counts_s.sum()

    # Return the TVD between P_proj_ebm and P_proj_sampler
    tvd = 0.5 * np.abs(p_proj_ebm - p_proj_sampler).sum()
    return tvd

def compute_naive_tvd_bounds(nreps, nbins, bin_func, scorer, p, q):
    tvds = []
    for _ in range(nreps): 
        s_q = q.sample()
        logp_sq = p.log_score(s_q)
        logq_sq = q.log_score(s_q)
        scored_sq = scorer.score(s_q)
        s = [x for (x, b) in zip(s_q, scored_sq) if b]
        tvd = compute_binned_tvd(s_q, logp_sq, logq_sq, s, bin_func, nbins)
        tvds.append(tvd)
    return tvds

def compute_imh_tvd_bounds(nreps, nbins, bin_func, scorer, p, q, sample_size, keep_every, burn_in):
    tvds = []
    for _ in range(nreps): 
        imh = IMHSampler(p, q, batch_size=sample_size, keep_every=keep_every, burn_in=burn_in)
        s_q = q.sample()
        logp_sq = p.log_score(s_q)
        logq_sq = q.log_score(s_q)
        s = imh.sample(return_diagnostics=False)
        tvd = compute_binned_tvd(s_q, logp_sq, logq_sq, s, bin_func, nbins)
        tvds.append(tvd)
    return tvds

def compute_bleu(tup):
    sample_idx, samples = tup
    # return nltk.translate.bleu_score.sentence_bleu(samples[:sample_idx] + samples[sample_idx+1:], samples[sample_idx], [1./3] * 3, smoothing_function=SmoothingFunction().method1)
    return  sacrebleu.sentence_bleu(samples[sample_idx], samples[:sample_idx] + samples[sample_idx+1:]).score

def self_bleu(samples):
    """
    My understanding of self-BLEU, a parallel implementation
    """
    tokenizer = nltk.word_tokenize 
    # tokenized_samples = [tokenizer(sample.rstrip()) for sample in samples]
    tokenized_samples = samples
    with multiprocessing.Pool(os.cpu_count()) as pool:
        bleu_scores = pool.map(compute_bleu, [(i, samples) for i in range(len(samples))])
    return np.average(bleu_scores)

if __name__ == "__main__":
    main()
