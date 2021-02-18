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
MARKER_SIZE=100


@st.cache(allow_output_mutation=True)
def load_samples(experiment):

    data_folder = Path(f"<output-dir>/{experiment}/")
    samples_file = data_folder / f"direct-{experiment}.pkl"
    ebm_scores_file = data_folder / f"direct-{experiment}.ebm.pkl"
    proposal_scores_file = data_folder / f"direct-{experiment}.proposal.pkl"
    base_scores_file = data_folder / f"direct-{experiment}.base.pkl"

    p = SampleRepository(samples_file, ebm_scores_file)
    q = SampleRepository(samples_file, proposal_scores_file)
    a = SampleRepository(samples_file, base_scores_file)
    return p, q, a

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

def main(samples_dir, figures_dir, df_dir):
    st.title("IMH vs QRS")
    sample_size = 1_000
    experiment = "female_science"

    # Load data
    P, q, a = load_samples(experiment)
    constraints = get_constraints(experiment)
    st.write(f"* Loaded {P.size:,} samples from disk.")

    # Gather IMH samples
    imh_settings = [(1000, 10), (1000, 100), (1000, 1000)] # (burn-in, thinning)
    imh_acceptance_rates = [sample_size / (sample_size * thinning + burn_in) for burn_in, thinning in imh_settings]
    samples = []
    samplers = []
    ars = []
    for burn_in, thinning in imh_settings:
        samples_file = samples_dir / f"IMH-{burn_in}-{thinning}-{sample_size}-{experiment}.npy"
        if samples_file.exists():
            sample = np.load(samples_file, allow_pickle=True)
        else:
            imh = IMHSampler(ebm=P, proposal=q, batch_size=sample_size, keep_every=thinning, burn_in=burn_in)
            sample = imh.sample()
            np.save(samples_file, samples)
        ar = sample_size / (sample_size * thinning + burn_in)
        samples.append(sample)
        samplers.append("IMH")
        ars.append(ar)
        

    # Gather QRS samples
    qrs_settings = [1e-1, 1e-2, 1e-3]
    qrs_acceptance_rates = qrs_settings
    for min_ar in qrs_settings:
        samples_file = samples_dir / f"QRS-{min_ar:.6f}-{sample_size}-{experiment}.npy"
        if samples_file.exists():
            sample = np.load(samples_file, allow_pickle=True).tolist()
        else:
            qrs = QuasiRejectionSampler(ebm=P, proposal=q, batch_size=sample_size, min_acceptance_rate=min_ar,
                                        silent=True)
            sample = qrs.sample()
            np.save(samples_file, sample)
        samples.append(sample)
        samplers.append("QRS")
        ars.append(min_ar)

    #  construct dataframe
    df = {
        "sampler": [],
        "ar": [],
        "ar_str": [],
        "p_uniq": [],
        #"ppl": []
    }
    for name, _, _ in constraints: df[f"constraint-{name}"] = []
    for sample, sampler, ar in zip(samples, samplers, ars):
        df["sampler"].append(sampler)
        df["ar"].append(ar)
        p_uniq = len(set(sample)) / len(sample)
        df["p_uniq"].append(p_uniq)
        for name, target, scorer in constraints:
            cs = np.mean(scorer.score(sample).data)
            df[f"constraint-{name}"].append(cs)
        #loga = a.log_score(sample)
        #lengths = [len(s.split()) for s in sample] # TODO
        #df["ppl"].append(np.exp(-(1.0 / sum(lengths)) * loga.sum()))
    st.write(df)

    for tgt_ar in qrs_settings * 2:
            df["ar_str"].append(f"{tgt_ar:.0e}")

    # Plot CS
    fig, axes = plt.subplots(len(constraints), 1)
    if len(constraints) == 1: axes = [axes]
    axes[0].set_title("Constraint Satisfaction")
    for constraint, ax in zip(constraints, axes):
        name = constraint[0]
        target = constraint[1]
        pdf = pd.DataFrame(df)
        sns.barplot(data=pdf, x="ar_str", y=f"constraint-{name}", hue="sampler", ax=ax)
        legend = ax.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
        ax.set_xlabel("acceptance rate")
        ax.set_ylabel(f"{name} %")
        ax.axhline(y=target, linestyle="dashed", color="black")
    fig.tight_layout()
    st.pyplot(fig)

    _ = """
    # Plot PPL
    fig, ax = plt.subplots()
    ax.set_title("Perplexity")
    pdf = pd.DataFrame(df)
    sns.barplot(data=pdf, x="ar_str", y="ppl", hue="sampler", ax=ax)
    legend = ax.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
    ax.set_xlabel("acceptance rate")
    ax.set_ylabel("perplexity")
    fig.tight_layout()
    st.pyplot(fig) 
    """

    # Plot uniq
    fig, ax = plt.subplots()
    ax.set_title("% unique sequences")
    pdf = pd.DataFrame(df)
    sns.barplot(data=pdf, x="ar_str", y="p_uniq", hue="sampler", ax=ax)
    legend = ax.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
    ax.set_xlabel("acceptance rate")
    ax.set_ylabel("% unique sequences")
    fig.tight_layout()
    st.pyplot(fig) 

    # Plot summary
    palette = sns.color_palette("colorblind")
    palette = palette[2:] + palette[:2]
    nplots = len(constraints) + 1
    fig, axes = plt.subplots(1, nplots, figsize=(nplots*5, 5))
    pdf = pd.DataFrame(df)
    ax = axes[0]
    pdf["p_uniq"] = pdf["p_uniq"].apply(lambda p: p*100.)
    sns.barplot(data=pdf, x="ar_str", y="p_uniq", hue="sampler", ax=ax, palette=palette)
    ax.set_xlabel("acceptance rate")
    ax.set_ylabel("% unique sequences")
    ax.get_legend().remove()
    for i, constraint in enumerate(constraints):
        ax = axes[i+1]
        name = constraint[0]
        target = constraint[1] * 100
        pdf[f"constraint-{name}"] = pdf[f"constraint-{name}"].apply(lambda cs: cs*100.)
        sns.barplot(data=pdf, x="ar_str", y=f"constraint-{name}", hue="sampler", ax=ax, palette=palette)
        #legend = ax.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
        ax.set_xlabel("acceptance rate")
        ax.set_ylabel(f"{name} %")
        ax.axhline(y=target, linestyle="dashed", label="target moments", color=colors[-3])
        ax.get_legend().remove()
    legend = ax.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
    fig.tight_layout()
    savefig = figures_dir / f"QRS-vs-IMH-{sample_size}.png"
    if savefig:
        fig.savefig(savefig, dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
    st.pyplot(fig) 


def plot_constraint_satisfaction(cs_dict, constraints, target_correction, observed_values=None, experiment=None, savefig=None, axes=None, legend_pos=None, show_ebm=True):
    df = pd.DataFrame(cs_dict)
    df["sampler"] = df["sampler"].apply(lambda sampler: f"{sampler} (SNIS)" if sampler == r"$p_\beta$" else f"{sampler}")
    fig = None
    if axes is None:
        fig, axes = plt.subplots(len(constraints), 1)
    #if len(constraints) == 1: axes = [axes]

    for constraint, ax in zip(constraints, axes):
        name, target, _ = constraint
        ebm = target_correction[name]
        graph = sns.lineplot(data=df, x="ar", y=f"constraint-{name}", ax=ax, style="sampler", hue="sampler", markers=False, dashes=False, legend=False)
        graph.axhline(target, linestyle="dashed", label="target moments", color=colors[-3])
        if show_ebm:
            graph.axhline(ebm, linestyle="dashed", label="p", color=colors[-4])
        sns.scatterplot(data=df, x="ar", y=f"constraint-{name}", ax=ax, style="sampler", hue="sampler", s=MARKER_SIZE)
        if observed_values:
            ar = observed_values[name]["ar"]
            cs = observed_values[name]["cs"]
            ax.scatter(ar, cs, marker='*', label=r"$p_\beta$ (observed)", color=colors[7], s=MARKER_SIZE)
        ax.set_xlabel("acceptance rate")
        ax.set(xscale='log')
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]+0.03)
        ax.invert_xaxis()
        if ax == axes[0]:
            if legend_pos is not None:
                legend = ax.legend()
            else:
                legend = ax.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
        else:
            ax.get_legend().remove()
    axes[0].set_title(experiment)

    if fig is not None: 
        if savefig:
            fig.savefig(savefig, dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')

        return fig

def plot_kl_a(kl_dict, experiment=None, savefig=None, ax=None, legend_pos=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    kl_df = pd.DataFrame(kl_dict)
    kl_df["sampler"] = kl_df["sampler"].apply(lambda sampler: fr"KL({sampler}||$a$)")
    sns.lineplot(data=kl_df, x="ar", y="kl_a", ax=ax, style="sampler", hue="sampler", markers=False, dashes=False, legend=False)
    sns.scatterplot(data=kl_df, x="ar", y="kl_a", ax=ax, style="sampler", hue="sampler", s=MARKER_SIZE)
    ax.set_xlabel("acceptance rate")
    ax.set(xscale='log')
    ax.set_ylabel("KL(.||a)")
    ax.set_title(experiment)
    ax.invert_xaxis()
    if legend_pos is not None:
        legend = ax.legend()
    else:
        legend = ax.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
    if fig is not None:
        if savefig:
            fig.savefig(savefig, dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
        return fig

    
def plot_kl_p(kl_dict, experiment=None, savefig=None, ax=None, legend_pos=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    kl_df = pd.DataFrame(kl_dict)
    kl_df["sampler"] = kl_df["sampler"].apply(lambda sampler: f"KL(p||{sampler})")
    sns.lineplot(data=kl_df, x="ar", y="kl_p", ax=ax, style="sampler", hue="sampler", markers=False, dashes=False, legend=False)
    sns.scatterplot(data=kl_df, x="ar", y="kl_p", ax=ax, style="sampler", hue="sampler", s=MARKER_SIZE)
    ax.set_xlabel("acceptance rate")
    ax.set(xscale='log')
    ax.set_ylabel("KL(p||.)")
    ax.set_title(experiment)
    ax.invert_xaxis()
    if legend_pos is not None:
        legend = ax.legend()
    else:
        legend = ax.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
    if fig is not None:
        if savefig:
            fig.savefig(savefig, dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
        return fig

def plot_tvd(qrs_ub, tvd_dict, experiment=None, savefig=None, ax=None, legend_pos=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    tvd_df = pd.DataFrame(tvd_dict)
    tvd_df["sampler"] = tvd_df["sampler"].apply(lambda sampler: f"TVD(p, {sampler})")
    qrs_ub["sampler"] = qrs_ub["ar"].apply(lambda _: "x")
    sns.lineplot(data=tvd_df, x="ar", y="tvd", ax=ax, style="sampler", hue="sampler", markers=False, dashes=False, legend=False)
    sns.scatterplot(data=tvd_df, x="ar", y="tvd", ax=ax, style="sampler", hue="sampler", s=MARKER_SIZE)
    sns.lineplot(data=qrs_ub, x="ar", y="tvd", ax=ax, label=r"$1 - p(A_\beta)$", color=colors[3])
    # sns.lineplot(data=qrs_ub, x="ar", y='tvd_with_bound', ax=ax,  label='QRS Hoeffding bound', color=colors[4])
    ax.set_ylabel('TVD')
    # ax.set(yscale='log')
    ax.set(xscale='log')
    ax.set_xlabel("acceptance rate")
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
        return fig
    
def qrs_upper_bound(logp_sq, logq_sq, multiplier=1):
    beta_x = np.exp(logp_sq - logq_sq)
    beta_tick_size = (beta_x.max() - beta_x.min()*multiplier) / 1000.
    beta_ticks = np.arange(beta_x.min(), beta_x.max()*multiplier, beta_tick_size)
    delta = 0.05
    df_data = get_tvd_estimates(beta_x, beta_ticks)
    df_data = add_bounds(df_data, beta_ticks, len(logp_sq), delta)
    return df_data
    
def plot_self_bleu(samplers, samples, N=[3, 4, 5]):
    fig, axes = plt.subplots(1, len(N), sharey=True)
    for n, ax in zip(N, axes):
        st.write(n)
        self_bleu = SelfBLEU(gram=n)
        df = {
            f"self-BLEU-{n}": [],
            "labels": [],
            "sampling method": []
        }
        for sampler_name, s in zip(samplers, samples):
            st.write(sampler_name)
            score = self_bleu.compute_metric(s)
            df[f"self-BLEU-{n}"].append(score)
            df["labels"].append(sampler_name)
            df["sampling method"].append("")
        df = pd.DataFrame(df)
        sns.barplot(x="sampling method", y=f"self-BLEU-{n}", hue="labels", data=df, ax=ax)
        ax.get_legend().remove()
    axes[-1].legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
    return fig

def plot_distinct_n(samplers, samples, N=3):
    fig, axes = plt.subplots(1, N, sharey=True)
    for n, ax in enumerate(axes):
        n += 1
        distn = Distinct_N(n)
        df = {
            f"Dist-{n}": [],
            "labels": [],
            "sampling method": []
        }
        for sampler_name, s in zip(samplers, samples):
            score = distn.compute_metric(s)
            df[f"Dist-{n}"].append(score)
            df["labels"].append(sampler_name)
            df["sampling method"].append("")
        df = pd.DataFrame(df)
        sns.barplot(x="sampling method", y=f"Dist-{n}", hue="labels", data=df, ax=ax)
        ax.get_legend().remove()
    axes[-1].legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
    return fig

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = Path(base_dir) / "samples"
    figures_dir = Path(base_dir) / "figures"
    df_dir = Path(base_dir) / "dataframes" 
    samples_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    df_dir.mkdir(exist_ok=True)
    main(samples_dir, figures_dir, df_dir)
