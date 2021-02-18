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

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from pathlib import Path
from cycler import cycler
from transformers import AutoTokenizer

from discontrol.sampler import QuasiRejectionSampler, IMHSampler
from discontrol.distribution.samplerepo import SampleRepository
from discontrol.scorer.common import broadcast
from discontrol.scorer import SingleWordFeature, GenderFeature, MultiWordFeature, ExponentialScorer
from discontrol.pipeline import build_sampler_from_config
from discontrol.metrics import Distinct_N, SelfBLEU
from discontrol.misc import NumpyArray

from qrs_tvd import  get_tvd_estimates, add_bounds, project_into_curve_x

# Plot settings
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
         '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
custom_cycler = cycler(color=colors)
plt.rc('axes', prop_cycle=custom_cycler)
sns.set_palette(sns.color_palette("colorblind"))
MARKER_SIZE=50

prompts = ["simple", "multiple", "knowledge", "jeopardy", "news-2", "start-wikileaks"]

@st.cache(allow_output_mutation=True)
def load_samples(experiment, proposal="direct"):
    # Hacky to accomodate the locations and namings of different experiments.

    if proposal == "base": proposal_extension = "base"
    else: proposal_extension = "proposal"
    
    name_extension = "" if proposal in prompts else f"-{experiment}"

    if proposal in prompts:
        data_folder = Path(f"<output-dir>/prompts/{experiment}/")
    else:
        data_folder = Path(f"<output-dir>/{experiment}/")

    samples_file = data_folder / f"{proposal}{name_extension}.pkl"
    ebm_scores_file = data_folder / f"{proposal}{name_extension}.ebm.pkl"
    proposal_scores_file = data_folder / f"{proposal}{name_extension}.{proposal_extension}.pkl"

    print(samples_file)
    p = SampleRepository(samples_file, ebm_scores_file)
    q = SampleRepository(samples_file, proposal_scores_file)
    return p, q

def get_constraints(experiment):
    if experiment == "amazing":
        return [("amazing", 1., SingleWordFeature("amazing"))]
    elif experiment == "female":
        return [("female", 0.5, GenderFeature("female"))]
    elif experiment == "female_science":
        return [("female", 0.5, GenderFeature("female")),
                ("science", 1., MultiWordFeature("resources/wikibio-wordlists/science.txt"))]
    elif experiment == "female_sports":
        return [("female", 0.5, GenderFeature("female")),
                ("sports", 1., MultiWordFeature("resources/wikibio-wordlists/sports.txt"))]
    elif experiment == "wikileaks":
        return [("wikileaks", 1., SingleWordFeature("Wikileaks"))]

def main(samples_dir, figures_dir, df_dir):
    st.title("Comparing Proposal Distributions")
    experiment = "wikileaks"
    st.write(f"# {experiment}")

    df = {
        "ar": [],
        "beta": [],
        "tvd": [],
        "sampler": [],
        "proposal": [],
    }
    constraints = get_constraints(experiment)
    for name, _, _ in constraints: df[f"constraint-{name}"] = []

    mixture_components = ["simple", "multiple", "knowledge", "jeopardy", "news-2", "start-wikileaks"]
    mixture_weights = [0.9054, 0.0038, 0.0458, 0.0136, 0.0243, 0.0071]
    
    mixture_sample = []
    mixture_logP = []
    mixture_logq = []
    mixture_betas = []
    df_proposal = {
        "ar": [],
        "tvd": [],
        "sampler": [],
        "proposal": []
    }
    for proposal in ["gdc", "base"] + prompts + ["mixture"]:

        if proposal != "mixture":
            # Load data
            P, q = load_samples(experiment, proposal=proposal)
            q.reset()
            q.batch_size = q.size #1_000_000 
            st.write(f"* Loaded {P.size:,} samples from disk.")

            # Collect a large sample for estimating betas.
            sample = q.sample()
            logP = P.log_score(sample).data
            logq = q.log_score(sample).data
            Z_P = np.mean(np.exp(logP - logq))
            logp = logP - np.log(Z_P) 

            betas = np.exp(logp - logq)
            multiplier = {
                "wikileaks": 0.01,
                "amazing": 0.01,
                "female_sports": 5000,
                "female_science": 1000,
                "female": 10}[experiment]
            betas = np.logspace(np.log(betas.min()+1e-12), np.log(betas.max()*multiplier), num=50, base=np.e)
            st.write(f"* {proposal}: Using a range of {len(betas)} betas ranging from {betas.min()+1e-12:.2g} to {betas.max()*multiplier:.2g}.")
        else:
            betas = np.logspace(np.log(min(mixture_betas)), np.log(max(mixture_betas)), num=50, base=np.e)

        # Use a new sample for estimating divergence metrics.
        if proposal == "mixture":
            sample = mixture_sample
            logP = np.array(mixture_logP)
            logq = np.array(mixture_logq)
            st.write(f"* Found {len(sample)} mixture samples")
        else:
            q.reset()
            sample = q.sample()
            logP = P.log_score(sample).data
            logq = q.log_score(sample).data
        Z_P = np.mean(np.exp(logP - logq))
        logp = logP - np.log(Z_P) 
        st.write(f"* Using {len(sample):,} samples for making importance-sampled estimates using q(x).")

        target_correction = {}

        # Score the sample
        scored_sample = {}
        for name, _, scorer in constraints:
            scored_sample[name] = scorer.score(sample).data

        # Mixture of Prompts
        if proposal in mixture_components:
            component_idx = mixture_components.index(proposal)
            weight = mixture_weights[component_idx]
            nsamples = int(np.ceil(q.batch_size * weight))
            mixture_sample.extend(sample[:nsamples])
            mixture_logP.extend(logP[:nsamples].tolist())
            mixture_logq.extend((logq[:nsamples] + np.log(weight)).tolist())
            mixture_betas.extend(betas)

        if proposal in ["start-wikileaks", "mixture"]: continue

        # add TVD of the raw proposal.
        tvd_proposal = 0.5 * np.mean(np.abs(1.0 - np.where(np.isneginf(logP), 0., np.exp(logp - logq))))
        df_proposal["ar"].append(1.0)
        df_proposal["tvd"].append(tvd_proposal)
        df_proposal["sampler"].append("q")
        proposal_name = proposal if proposal != "gdc" else "DPG"
        proposal_name = proposal_name if proposal_name != "base" else "GPT-2 small"
        df_proposal["proposal"].append(proposal_name)

        cutoffs = {
                'gdc': 1e-3,
                'multiple': 4e-5,
                'news': 4e-5,
                'simple': 8e-5
                }

        if proposal in prompts + ['gdc']:
            if proposal == 'gdc':
                data_folder = Path(f"<output-dir>/1M/{experiment}/")
                proposal_df = np.load(data_folder / f'direct-wikileaks.estimates.pkl', allow_pickle=True)
            else:
                data_folder = Path(f"<output-dir>/prompts/{experiment}/")
                proposal_df = np.load(data_folder / f'{proposal}.estimates.pkl', allow_pickle=True)
            proposal_df = proposal_df[proposal_df['sampler'] == 'qrs']
            proposal_df = proposal_df[proposal_df['estimator'] == 'bootstrap']
            if proposal in cutoffs:
                proposal_df = proposal_df[proposal_df['ar'] > cutoffs[proposal]]
            else:
                proposal_df = proposal_df[proposal_df['ar'] > 1e-5]
            #proposal_df = proposal_df.groupby('beta').mean()
            proposal_df['proposal'] = proposal if proposal != 'gdc' else 'DPG'
            proposal_df['sampler'] = r'$p_\beta$'
            proposal_df['constraint-wikileaks'] = proposal_df['constraint-Wikileaks']
            for k,v in proposal_df.to_dict('list').items():
                if k in df:
                    df[k].extend(v)
        else:
            # Compute SNIS estimates using q.
            for beta in betas:
                r_x = np.where(np.isneginf(logP), 0., np.minimum(1, np.exp(logP - np.log(beta) - logq)))
                Z_K = np.mean(r_x)
                if Z_K < 1e-5: break # acceptance rates lower than 1e-5 are not trustworthy for 1e6 samples.
                logK = np.minimum(logq, logP - np.log(beta))
                logk = logK - np.log(Z_K)
                tvd = 0.5 * np.mean(np.where(np.isneginf(logP), 0., np.abs(np.exp(logk - logq) - np.exp(logp - logq))))
                kl_p = np.mean(np.where(np.isneginf(logP), 0., np.exp(logp - logq) * (logp - logk)))
                for name, target, scorer in constraints:
                    cs = np.mean(np.exp(logk - logq) * scored_sample[name])
                    df[f"constraint-{name}"].append(cs)
                    if name not in target_correction:
                        cs = np.mean(np.exp(logp - logq) * scored_sample[name])
                        target_correction[name] = cs

                df['beta'].append(beta)
                df["ar"].append(Z_K)
                df["tvd"].append(tvd)
                df["sampler"].append(r"$p_\beta$")
                df["proposal"].append(proposal_name)
                if tvd < 1e-2: break # stop when TVD near zero



    # Total Variation Distance
    st.write("### Total Variation Distance")
    savefig = figures_dir / f"QRS-TVD-prompting-{experiment}-{q.batch_size}.png"
    fig, ax = plot_tvd(df, experiment=None, savefig=savefig, legend_pos="inside")
    
    df_proposal = pd.DataFrame(df_proposal)
    df_proposal["proposal"] = df_proposal["proposal"].apply(lambda proposal: proposal if proposal != "news-2" else "news")
    sns.scatterplot(data=df_proposal, x="ar", y="tvd", ax=ax, style="proposal", hue="proposal", s=MARKER_SIZE, legend=False)
    st.write(df_proposal)
    ax.set_ylabel('TVD')
    ax.set_xlabel("acceptance rate")

    fig.savefig(savefig, dpi=300, bbox_inches='tight')
    st.pyplot(fig)

from pandas.api.types import is_numeric_dtype

def mean_str(col):
    if is_numeric_dtype(col):
        return col.mean()
    else:
        return col.unique() if col.nunique() == 1 else np.NaN

def plot_tvd(tvd_dict, experiment=None, savefig=None, ax=None, legend_pos=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    tvd_df = pd.DataFrame(tvd_dict)
    tvd_df["proposal"] = tvd_df["proposal"].apply(lambda proposal: proposal if proposal != "news-2" else "news")
    tvd_df['binned_ar'] = tvd_df.groupby(['proposal','beta'])['ar'].transform('mean')
    tvd_df = tvd_df.sort_values('proposal')
    sns.lineplot(data=tvd_df, x="binned_ar", y="tvd", ax=ax, style="proposal", hue="proposal", ci='sd', err_style='band', markers=False, dashes=False, legend=False)
    mean_df = tvd_df.groupby(['proposal', 'binned_ar']).agg(mean_str).reset_index()
    sns.scatterplot(data=mean_df, x="binned_ar", y="tvd", ax=ax, style="proposal", hue="proposal", s=MARKER_SIZE)

    # improve visibility GPT-2 small
    gpt2_df = tvd_df[tvd_df["proposal"] == "GPT-2 small"]
    sns.scatterplot(data=gpt2_df, x="binned_ar", y="tvd", ax=ax, s=MARKER_SIZE, legend=False, marker="X", zorder=100)

    ax.set_ylabel('TVD')
    ax.set(xscale='log')
    ax.set_xlabel("acceptance rate")
    ax.set_xlim(xmin=3e-8, xmax=None)
    ax.invert_xaxis()
    if legend_pos is not None:
        legend = ax.legend()
    else:
        legend = ax.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
    if experiment:
        ax.set_title(experiment)
    if fig is not None:
        if savefig:
            fig.savefig(savefig, dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
            st.write(f'Figure saved to {savefig}')
        return fig, ax
    
def qrs_upper_bound(logp_sq, logq_sq, multiplier=1):
    beta_x = np.exp(logp_sq - logq_sq)
    beta_tick_size = (beta_x.max() - beta_x.min()*multiplier) / 1000.
    beta_ticks = np.arange(beta_x.min(), beta_x.max()*multiplier, beta_tick_size)
    delta = 0.05
    df_data = get_tvd_estimates(beta_x, beta_ticks)
    df_data = add_bounds(df_data, beta_ticks, len(logp_sq), delta)
    return df_data

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = Path(base_dir) / "samples"
    figures_dir = Path(base_dir) / "figures"
    df_dir = Path(base_dir) / "dataframes" 
    samples_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    df_dir.mkdir(exist_ok=True)
    main(samples_dir, figures_dir, df_dir)
