# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from cycler import cycler

#from transformers import MarianMTModel, MarianTokenizer

from discontrol.scorer import KeywordsFeature, ParaphraseFeature, ProductScorer
from discontrol.distribution.nmt import FBRoundTripNMT
from discontrol.distribution.lm import HFAutoregressiveLM
from discontrol.sampler import QuasiRejectionSamplerFixedBeta
from discontrol.distribution.samplerepo import SampleRepository

from pathlib import Path

from stqdm import stqdm

# Plot settings
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
         '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
custom_cycler = cycler(color=colors)
plt.rc('axes', prop_cycle=custom_cycler)
sns.set_palette(sns.color_palette("colorblind"))
MARKER_SIZE=100

@st.cache(allow_output_mutation=True)
def load_samples(seq_idx):
    base_dir = Path(f"<output-dir>/paraphrasing-95/{seq_idx}/")
    samples_file = base_dir / "round-trip-nmt.npy"
    q = SampleRepository(samples_file, base_dir / "round-trip-nmt.proposal.npy")
    P_sup = SampleRepository(samples_file, base_dir / "round-trip-nmt.ebm-sup.npy")
    P_unsup = SampleRepository(samples_file, base_dir / "round-trip-nmt.ebm-unsup.npy")
    feature_sup = np.load(base_dir / "round-trip-nmt.feature-sup.npy")
    feature_unsup = np.load(base_dir / "round-trip-nmt.feature-unsup.npy")
    return q, feature_unsup, P_unsup, feature_sup, P_sup

def main(figures_dir):
    st.set_page_config(layout="wide") 
    st.title("Paraphrase Generation")

    orig = [
        'How is the two wheeler insurance from Bharti Axa insurance?',
        'Are there Doctor Who references in the Muse song "Knights of Cydonia"?',
        'In French, how do you say "cool"?'
    ]

    plot_titles = [
        'How is the two wheeler insurance from\nBharti Axa insurance?',
        'Are there Doctor Who references in the Muse song\n"Knights of Cydonia"?',
        'In French, how do you say "cool"?'
    ]

    fig, axes = plt.subplots(1, len(orig), figsize=(15, 2), sharex=False, sharey=True)

    for seq_idx in range(0, 3):
        input_seq = orig[seq_idx]
        q, feature_unsup, P_unsup, feature_sup, P_sup = load_samples(seq_idx)
        q.reset()

        st.write(f"### Input Sequence: \"{input_seq}\"")
        st.write(f"* constraint satisfaction = {feature_sup.mean():.1%}")

        nsamples = 1_000_000
        q.batch_size = nsamples

        betas_sup = np.logspace(np.log(1e-40), np.log(1e-15), num=50, base=np.e)
        if seq_idx == 2:
            betas_sup = np.logspace(np.log(1e-11), np.log(1e-4), num=50, base=np.e)

        betas_unsup = betas_sup

        # Collect a new sample for estimating TVD
        q.reset()
        sample = q.sample()
        logq = q.log_score(sample).data

        logP_sup = P_sup.log_score(sample).data
        ZP_sup = np.mean(np.exp(logP_sup - logq))
        logp_sup = logP_sup - np.log(ZP_sup) 


        
        # Compute TVD
        N  = len(logq)
        st.write(f"* Using {N:,} samples for making importance-sampled estimates with q(x).")
        N_BOOTSTRAP = 100
        df = compute_metrics_n_times(logP_sup, logp_sup, logq, betas_sup, 
                [np.arange(N)])
        unit_line = np.logspace(np.log(df['ar'].min()), 1e-10, base=np.e, endpoint=True)
        df['binned_ar'] = unit_line[np.digitize(df['ar'], unit_line, right=True)]#pd.cut(df_data['ar'], 1000, right=False)
        df_bootstrap = compute_metrics_n_times(logP_sup, logp_sup, logq, betas_sup, 
                [np.random.choice(N, N) for i in range(N_BOOTSTRAP)])
        df_bootstrap['binned_ar'] = unit_line[np.digitize(df_bootstrap['ar'], unit_line, right=True)]#pd.cut(df_data['ar'], 1000, right=False)
        st.write(df_bootstrap)
        mean_tvd = df.groupby('beta')['TVD'].mean().rename('TVD_mean')
        mean_ar = df.groupby('beta')['ar'].mean().rename('ar')
        std_tvd = df_bootstrap.groupby('beta')['TVD'].std().rename('TVD_std')
        df = pd.concat([mean_ar, mean_tvd, std_tvd], axis=1).reset_index().dropna()
        #st.write(df)
        
        #fig, ax = plt.subplots()
        ax = axes[seq_idx]
        ax.set_facecolor("white")
        color = sns.color_palette("colorblind")[2]
        sns.set_style("ticks")
        ax.errorbar(df['ar'], df['TVD_mean'], yerr=df['TVD_std'], color=color, marker='o')
        ax.set(xscale='log')
        ax.set_ylabel("TVD")
        ax.set_xlabel("acceptance rate")
        lower, upper = ax.get_ylim()
        ax.set_ylim(bottom=-0.07, top=1.03*upper) 
        ax.invert_xaxis()
        ax.set_title(plot_titles[seq_idx])
        ax.yaxis.set_major_formatter('{x:0<4.2f}')
        ax.yaxis.set_tick_params(labelbottom=True)
        if seq_idx > 0:
            ax.set_ylabel("")

        st.write(df)
    
    fig.savefig(figures_dir / f"QRS-paraphrasing-{nsamples}.png", dpi=300, bbox_inches='tight')
    st.pyplot(fig)

def compute_metrics_n_times(logP_sup, logp_sup, logq, betas, xis):
    df = {
        "EBM": [],
        "ar": [],
        "TVD": [],
        "beta": []
    }
    for xi in stqdm(xis, desc='Bootstrap simulation'):
        for beta in betas: 
            logP_sup_xi = logP_sup[xi]
            logp_sup_xi = logp_sup[xi]
            logq_xi = logq[xi]
            compute_metrics(df, beta, logP_sup_xi, logp_sup_xi, logq_xi)
    return pd.DataFrame(df)

def compute_metrics(df, beta, logP_sup_xi, logp_sup_xi, logq_xi):
    ZP_sup = np.mean(np.exp(logP_sup_xi - logq_xi))
    logp_sup_xi = logP_sup_xi - np.log(ZP_sup) 
    r_x = np.where(np.isneginf(logP_sup_xi), 0., np.minimum(1, np.exp(logP_sup_xi - np.log(beta) - logq_xi)))
    ZK = np.mean(r_x)
    logK = np.minimum(logq_xi, logP_sup_xi - np.log(beta))
    logk = logK - np.log(ZK)
    tvd = min(0.5 * np.mean(np.where(np.isneginf(logP_sup_xi), 0., np.abs(np.exp(logk - logq_xi) - np.exp(logp_sup_xi - logq_xi)))), 1)
    if ZK < 5e-6: return
    df["EBM"].append("supervised") 
    df["TVD"].append(tvd) 
    df["ar"].append(ZK)
    df["beta"].append(beta)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = Path(base_dir) / "figures"
    main(figures_dir)
