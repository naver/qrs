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

def load_scores(experiment):
    proposal = "direct" if experiment != "wikileaks" else "gdc"

    data_folder = Path(f"<output-dir>/{experiment}/")
    ebm_scores_file = data_folder / f"{proposal}-{experiment}.ebm.pkl"
    proposal_scores_file = data_folder / f"{proposal}-{experiment}.proposal.pkl"
    base_scores_file = data_folder / f"{proposal}-{experiment}.base.pkl"

    P = np.load(ebm_scores_file, allow_pickle=True)
    q = np.load(proposal_scores_file, allow_pickle=True)
    return P, q

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

def get_betas(experiment, num=10):
    if experiment == "female_science":
        beta_min = 1e-12
        beta_max = 9.3e6
    return np.logspace(np.log(beta_min), np.log(beta_max), num=num, base=np.e)

def main(samples_dir, figures_dir, df_dir):
    # sample_size = st.selectbox("sample size qrs samplers", [50_000, 100])
    experiment = "amazing"
    P, q = load_scores(experiment)
    sample_size = 1_000_000
    runs = 10
    assert len(P) >= sample_size * runs and len(q) >= sample_size * runs
    
    betas = [0.0301, 2.46]# female_science: get_betas(experiment, num=50).tolist() + [7.51e3, 9.21e5]
    ars = [[] for _ in range(len(betas))]
    tvds = [[] for _ in range(len(betas))]
    kls = [[] for _ in range(len(betas))]
    idx = 0
    for _ in range(runs):
        logP = P[idx:idx+sample_size]
        logq = q[idx:idx+sample_size]
       
        Z_P = np.mean(np.exp(logP - logq))
        logp = logP - np.log(Z_P)
        
        for j, beta in enumerate(betas):
            r_x = np.where(np.isneginf(logP), 0., np.minimum(1, np.exp(logP - np.log(beta) - logq)))
            Z_K = np.mean(r_x)
            logK = np.minimum(logq, logP - np.log(beta))
            logk = logK - np.log(Z_K)
            tvd = 0.5 * np.mean(np.where(np.isneginf(logP), 0., np.abs(np.exp(logk - logq) - np.exp(logp - logq))))
            kl_p = np.mean(np.where(np.isneginf(logP), 0., np.exp(logp - logq) * (logp - logk)))
            ars[j].append(Z_K)
            tvds[j].append(tvd)
            kls[j].append(kl_p)

        idx += sample_size 
    
    for j, beta in enumerate(betas):
        print(f"== beta: {beta:.1e}")
        print(f"AR: {np.mean(ars[j]):.1e} +- {np.std(ars[j]):.1e}")
        print(f"TVD: {np.mean(tvds[j])} +- {np.std(tvds[j])}")
        print(f"KL: {np.mean(kls[j])} +- {np.std(kls[j])}")
        print()

def plot_constraint_satisfaction(cs_dict, constraints, target_correction, observed_values=None, experiment=None, savefig=None, axes=None, legend_pos=None, show_ebm=True):
    df = pd.DataFrame(cs_dict)
    df["sampler"] = df["sampler"].apply(lambda sampler: f"{sampler} (SNIS)" if sampler == r"$p_\beta$" else f"{sampler}")
    fig = None
    if axes is None:
        fig, axes = plt.subplots(len(constraints), 1)
    # if len(constraints) == 1: axes = [axes]

    for constraint, ax in zip(constraints, axes):
        name, target, _ = constraint
        df[f"constraint-{name}"] = df[f"constraint-{name}"] * 100.
        target *= 100
        ebm = target_correction[name] * 100.
        graph = sns.lineplot(data=df, x="ar", y=f"constraint-{name}", ax=ax, style="sampler", hue="sampler", markers=False, dashes=False, legend=False)
        graph.axhline(target, linestyle="dashed", label="target moments", color=colors[-3])
        if show_ebm:
            graph.axhline(ebm, linestyle="dashed", label="p", color=colors[-4])
        sns.scatterplot(data=df, x="ar", y=f"constraint-{name}", ax=ax, style="sampler", hue="sampler", s=MARKER_SIZE)
        if observed_values:
            ar = observed_values[name]["ar"]
            cs = np.array(observed_values[name]["cs"]) * 100.
            ax.scatter(ar, cs, marker='*', label=r"$p_\beta$ (observed)", color=colors[7], s=MARKER_SIZE)
        ax.set_xlabel("acceptance rate")
        ax.set_ylabel(f"{name} %")
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
    ax.yaxis.set_major_formatter('{x:0<4.1f}')
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
    
def qrs_upper_bound(p_over_q, beta_ticks):
    df_data = get_tvd_estimates(p_over_q, beta_ticks)
    return df_data
    
def plot_self_bleu(samplers, samples, N=[3, 4, 5]):
    fig, axes = plt.subplots(1, len(N), sharey=True)
    for n, ax in zip(N, axes):
        self_bleu = SelfBLEU(gram=n)
        df = {
            f"self-BLEU-{n}": [],
            "labels": [],
            "sampling method": []
        }
        for sampler_name, s in zip(samplers, samples):
            score = self_bleu.compute_metric(s.data)
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
    samples_dir = Path("<output-dir>/gdc-qrs")
    figures_dir = Path(base_dir) / "figures"
    df_dir = Path(base_dir) / "dataframes" 
    samples_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    df_dir.mkdir(exist_ok=True)
    main(samples_dir, figures_dir, df_dir)
