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
    proposal = "direct" if experiment != "wikileaks" else "gdc"

    data_folder = Path(f"<output-dir>/1M/{experiment}/")
    samples_file = data_folder / f"{proposal}-{experiment}.pkl"
    ebm_scores_file = data_folder / f"{proposal}-{experiment}.ebm.pkl"
    proposal_scores_file = data_folder / f"{proposal}-{experiment}.proposal.pkl"
    base_scores_file = data_folder / f"{proposal}-{experiment}.base.pkl"

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
    elif experiment == "wikileaks":
        return [("wikileaks", 1., SingleWordFeature("Wikileaks"))]

def main(samples_dir, figures_dir, df_dir):
    st.title("GDC vs QRS")
    # sample_size = st.selectbox("sample size qrs samplers", [50_000, 100])
    experiment = st.selectbox("experiment", ["amazing", "female_sports", "female_science", "female", "wikileaks"])
    sample_size = 1000

    # Load data
    P, q, a = load_samples(experiment)
    q.reset()
    q.batch_size = 1_000_000
    constraints = get_constraints(experiment)
    st.write(f"* Loaded {P.size:,} samples from disk.")

    nreps = 1
    for rep_idx in range(nreps):
        df_file = df_dir / f"df-{experiment}-{q.batch_size}-{rep_idx}.npy"
        ub_file = df_dir / f"ub-{experiment}-{q.batch_size}-{rep_idx}.npy"

        # Collect a large sample for estimating betas.
        sample = q.sample()
        logP = P.log_score(sample).data
        logq = q.log_score(sample).data
        loga = a.log_score(sample).data
        Z_P = np.mean(np.exp(logP - logq))
        logp = logP - np.log(Z_P) 
        betas = np.exp(logp - logq)
        multiplier = {
            "wikileaks": 0.001,
            "amazing": 0.01,
            "female_sports": 5000,
            "female_science": 1000,
            "female": 10}[experiment]
        # FSP: 5000 FSC: 1000 F: 10, A: 0.01
        # Quickfix for getting a consistent range of betas for plotting
        betas = np.logspace(np.log(betas.min()+1e-12), np.log(betas.max()*multiplier), num=50, base=np.e)
        st.write(f"* Using a range of {len(betas)} betas ranging from {betas.min():.2e} to {betas.max():.1e}.")
        st.write(f"* Using a range of {len(betas)} betas ranging from {betas.min():.2f} to {betas.max():.2f}.")

        # Use a new sample for estimating divergence metrics.
        sample = q.sample()
        logP = P.log_score(sample).data
        logq = q.log_score(sample).data
        loga = a.log_score(sample).data
        Z_P = np.mean(np.exp(logP - logq))
        logp = logP - np.log(Z_P) 
        st.write(f"* Using {len(sample):,} samples for making importance-sampled estimates using q(x).")

        df = {
            "ar": [],
            "tvd": [],
            "kl_a": [],
            "kl_p": [],   
            "sampler": [],
        }
        for name, _, _ in constraints: df[f"constraint-{name}"] = []
        target_correction = {}

        # Score the sample
        scored_sample = {}
        for name, _, scorer in constraints:
            scored_sample[name] = scorer.score(sample).data

        # Compute SNIS estimates using q.
        for beta in betas:
            r_x = np.where(np.isneginf(logP), 0., np.minimum(1, np.exp(logP - np.log(beta) - logq)))
            Z_K = np.mean(r_x)
            logK = np.minimum(logq, logP - np.log(beta))
            logk = logK - np.log(Z_K)
            tvd = 0.5 * np.mean(np.where(np.isneginf(logP), 0., np.abs(np.exp(logk - logq) - np.exp(logp - logq))))
            kl_p = np.mean(np.where(np.isneginf(logP), 0., np.exp(logp - logq) * (logp - logk)))
            kl_a = np.mean(np.where(np.isneginf(logP), 0., np.exp(logk - logq) * (logk-loga)))
            for name, target, scorer in constraints:
                cs = np.mean(np.exp(logk - logq) * scored_sample[name])
                df[f"constraint-{name}"].append(cs)
                if name not in target_correction:
                    cs = np.mean(np.exp(logp - logq) * scored_sample[name])
                    target_correction[name] = cs

            df["ar"].append(Z_K)
            df["tvd"].append(tvd)
            df["kl_p"].append(kl_p)
            df["kl_a"].append(kl_a)
            df["sampler"].append(r"$p_\beta$")

        # Add GDC.
        tvd_p_gdc = 0.5 * np.mean(np.abs(1. - np.exp(logp - logq)))
        kl_p_gdc = np.mean(np.where(np.isneginf(logP), 0., np.exp(logp - logq) * (logp - logq)))
        kl_gdc_a = np.mean(logq - loga)
        df["ar"].append(1.)
        df["tvd"].append(tvd_p_gdc)
        df["kl_p"].append(kl_p_gdc)
        df["kl_a"].append(kl_gdc_a)
        df["sampler"].append("q")
        for name, _, scorer in constraints:
            df[f"constraint-{name}"].append(np.mean(scored_sample[name]))

        # Add naive filter
        if experiment in ["amazing", "wikileaks"]:
            ar = np.mean(scored_sample[experiment])
            log_NF = np.where(np.isneginf(logP), -np.infty, logq)
            Z_NF = np.mean(np.exp(log_NF - logq))
            log_nf = log_NF - np.log(Z_NF)
            tvd = 0.5 * np.mean(np.where(np.isneginf(logP), 0., np.abs(np.exp(log_nf - logq) - np.exp(logp - logq))))
            kl_p = np.mean(np.where(np.isneginf(logP), 0., np.exp(logp - logq) * (logp - log_nf)))
            kl_a = np.mean(np.where(np.isneginf(log_NF), 0., np.exp(log_nf - logq) * (log_nf - loga)))
            df["ar"].append(ar)
            df["tvd"].append(tvd)
            df["kl_p"].append(kl_p)
            df["kl_a"].append(kl_a)
            df["sampler"].append(r"$q_{\mathrm{proj}}$")
            df[f"constraint-{experiment}"].append(1.)

        # # Save dataframe.
        qrs_ub = get_tvd_estimates(np.exp(logP - logq), betas)

        ### Divergence Metrics
        st.write("## Divergence Metrics")

        # Forward KL to base.
        st.write("### KL to GPT-2")
        savefig = figures_dir / f"{experiment}-KL-a-{rep_idx}.png"
        savefig = None
        fig = plot_kl_a(df, experiment=experiment, savefig=savefig)
        st.pyplot(fig)

        # Total Variation Distance
        st.write("### Total Variation Distance")
        savefig = figures_dir / f"{experiment}-TVD-{rep_idx}.png"
        savefig = None
        fig = plot_tvd(qrs_ub, df, experiment=experiment, savefig=savefig)
        st.pyplot(fig)

        # Forward KL from the EBM.
        st.write("## KL from the EBM")
        savefig = figures_dir / f"{experiment}-KL-p-{rep_idx}.png"
        savefig = None
        fig = plot_kl_p(df, experiment=experiment, savefig=savefig)
        st.pyplot(fig)

        ## Theoretical Constraint Satisfaction

        ### Collect samples from QRS.
        st.write("## Running the QRS sampler")
        
        # QRS
        samplers = []
        samples = []
        acceptance_rates = [1e-1, 1e-2, 1e-3]
        for ar in acceptance_rates:
            samples_file = samples_dir / f"QRS-{ar:.6f}-50000-{experiment}.npy"
            if samples_file.exists():
                s = np.load(samples_file, allow_pickle=True)
                st.write(f"* Loaded {len(s)} samples for QRS@{ar}")
            else:
                raise NotImplementedError
            samplers.append(f"QRS@{ar:.0e}")
            samples.append(s)

        # Observed Constraint Satisfaction
        st.write("## Observed Constraint Satisfaction")
        st.write(f"* using {sample_size} samples.")
        observed_cs = {}
        for name, _, scorer in constraints:
            observed_cs[name] = {}
            cs = []
            for sampler_name, s in zip(samplers, samples):
                cs.append(np.mean(scorer.score(s[:sample_size]).data))
            observed_cs[name]["ar"] = acceptance_rates
            observed_cs[name]["cs"] = cs

        # Summary plot
        st.write("## Summary")
        savefig = figures_dir / f"QRS-{experiment}-{q.batch_size}.png"
        nplots = 3 + len(constraints)
        fig, axes = plt.subplots(1, nplots, figsize=(nplots*5, 5), sharex=False, sharey=False)
        plot_tvd(qrs_ub, df, ax=axes[0], legend_pos="")
        plot_kl_p(df, ax=axes[1], legend_pos="")
        show_ebm = experiment not in ["amazing", "wikileaks"]
        plot_constraint_satisfaction(df, constraints, target_correction, observed_values=observed_cs, axes=axes[2:2+len(constraints)], legend_pos="", show_ebm=show_ebm)
        plot_kl_a(df, ax=axes[-1], legend_pos="")
        if savefig:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')
        st.pyplot(fig)
        st.write(df)
        st.write([float(x) for x in df["constraint-female"]])
        st.write([float(x) for x in df["constraint-science"]])

        savefig = figures_dir / f"QRS-{experiment}-{q.batch_size}-nocs.png"
        nplots = 3 
        fig, axes = plt.subplots(1, nplots, figsize=(nplots*5, 2.5), sharex=False, sharey=False)
        plot_tvd(qrs_ub, df, ax=axes[0], legend_pos="")
        plot_kl_p(df, ax=axes[1], legend_pos="")
        plot_kl_a(df, ax=axes[-1], legend_pos="")
        if savefig:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')
        st.pyplot(fig)

        savefig = figures_dir / f"QRS-{experiment}-{q.batch_size}-cs.png"
        nplots = len(constraints)
        fig, axes = plt.subplots(1, nplots, figsize=(nplots*5, 2.5), sharex=False, sharey=False)
        show_ebm = experiment not in ["amazing", "wikileaks"]
        plot_constraint_satisfaction(df, constraints, target_correction, observed_values=observed_cs, axes=axes, legend_pos="", show_ebm=show_ebm)
        if savefig:
            fig.savefig(savefig, dpi=300, bbox_inches='tight')
        st.pyplot(fig)

def plot_constraint_satisfaction(cs_dict, constraints, target_correction, observed_values=None, experiment=None, savefig=None, axes=None, legend_pos=None, show_ebm=True):
    df = pd.DataFrame(cs_dict)
    df["sampler"] = df["sampler"].apply(lambda sampler: f"{sampler} (SNIS)" if sampler == r"$p_\beta$" else f"{sampler}")
    fig = None
    if axes is None:
        fig, axes = plt.subplots(len(constraints), 1)

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
    ax.set_ylabel('TVD')
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
