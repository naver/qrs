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
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from pathlib import Path

from discontrol.sampler import QuasiRejectionSampler, IMHSampler
from discontrol.distribution.poisson import PoissonDistribution
from discontrol.scorer import SingleWordFeature, GenderFeature, MultiWordFeature
from discontrol.distribution.samplerepo import SampleRepository

from gdc_vs_qrs import get_constraints

# Plot settings
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
         '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
sns.set_palette(sns.color_palette("colorblind"))

def main():
    st.title("Prompting for Wikileaks")

    #sample_dir = Path("<output-dir>/prompts/amazing/")
    sample_dir = Path("<output-dir>/prompts/wikileaks/")
    sample_files = ["simple.pkl", "multiple.pkl", "knowledge.pkl", "assange.pkl", "assange-full.pkl", "jeopardy.pkl", "news-1.pkl", "news-2.pkl", "now-to.pkl", "start-wikileaks.pkl", "this-is.pkl"]
    proposal_files = [f"{fn.split('.')[0]}.proposal.pkl" for fn in sample_files]
    ebm_files = [f"{fn.split('.')[0]}.ebm.pkl" for fn in sample_files]
    prompt_names = [fn.split('.')[0] for fn in sample_files]
    
    # Add unconditional
    sample_files += ["../../amazing/direct.pkl"]
    proposal_files += ["../../amazing/direct.base.pkl"]
    ebm_files += ["../../amazing/direct.ebm.pkl"]
    prompt_names += ["unconditional"]
       
    scorer = SingleWordFeature("Wikileaks")

    df = {
        "prompt": [],
        "constraint satisfaction": [],
        "TVD": [],
        "TVD-NF": []
    }

    def compute_stats(sample, log_P, log_p, log_q):
        # CS
        scored_sample = scorer.score(sample).data
        cs = np.mean(scored_sample)
        df["constraint satisfaction"].append(cs)

        # TVD
        tvd = 0.5 * np.mean(np.abs(1.0 - np.where(np.isneginf(log_P), 0., np.exp(log_p - log_q))))
        st.write(f"TVD: {tvd:.6f}")
        df["TVD"].append(tvd)

        # Naive Filter
        log_NF = np.where(np.isneginf(log_P), -np.infty, log_q)
        Z_NF = np.mean(np.exp(log_NF - log_q))
        log_nf = log_NF - np.log(Z_NF)
        # st.write(f"Z_P = {Z_P}")
        
        # TVD (NF)
        tvd_nf = 0.5 * np.mean(np.where(np.isneginf(log_P), 0., np.abs(np.exp(log_nf - log_q) - np.exp(log_p - log_q))))
        st.write(f"TVD_nf: {tvd_nf:.6f}")
        df["TVD-NF"].append(tvd_nf)

    only_show_mixture = True
    mixture_components = ["simple", "multiple", "knowledge", "jeopardy", "news-2", "start-wikileaks"] 
    mixture_weights = [0.9054, 0.0038, 0.0458, 0.0136, 0.0243, 0.0071]
    st.write(sum(mixture_weights))
    mixture_samples = []
    mixture_logP = []
    mixture_logq = []
    for _ in range(1):
        
        for prompt_name, sample_file, proposal_file, ebm_file in zip(prompt_names, sample_files, proposal_files, ebm_files):
            if only_show_mixture and prompt_name not in mixture_components: continue
            st.write(f"#### {sample_file}")
            df["prompt"].append(prompt_name)
            P = SampleRepository(sample_dir / sample_file, sample_dir / ebm_file)
            q = SampleRepository(sample_dir / sample_file, sample_dir / proposal_file)
            q.batch_size = 50_000
            sample = q.sample()

            # Z_P
            log_P = P.log_score(sample).data
            log_q = q.log_score(sample).data
            Z_P = np.mean(np.exp(log_P - log_q))
            log_p = log_P - np.log(Z_P)

            compute_stats(sample, log_P, log_p, log_q) 

            # Mixture of Prompts
            if prompt_name in mixture_components:
                component_idx = mixture_components.index(prompt_name)
                weight = mixture_weights[component_idx]
                nsamples = int(np.ceil(q.batch_size * weight))
                mixture_samples.extend(sample[:nsamples])
                mixture_logP.extend(log_P[:nsamples].tolist())
                mixture_logq.extend((log_q[:nsamples] + np.log(weight)).tolist())
                
                if len(mixture_samples) >= q.batch_size:
                    st.write("#### Mixture")
                    st.write(mixture_weights)
                    df["prompt"].append("mixture") 
                    log_P = np.array(mixture_logP)
                    log_q = np.array(mixture_logq)
                    Z_P = np.mean(np.exp(log_P - log_q))
                    log_p = log_P - np.log(Z_P)
                    compute_stats(mixture_samples, log_P, log_p, log_q)

    # Print prompts.
    prompts = {}
    config_folder = Path("/home/beikema/samplers/discontrol/experiments/config/proposals/prompts/wikileaks/")
    for prompt_name in prompt_names:
        if only_show_mixture and prompt_name not in mixture_components: continue
        if prompt_name == "unconditional": continue
        cfg = yaml.load(open(config_folder / f"{prompt_name}.yml"), Loader=Loader)
        prompts[prompt_name] = cfg["proposal"]["args"]["prompt"]
    st.write(prompts)

    # Plot CS and TVD.
    fig, axes = plt.subplots(3, 1, figsize=(12, 36))
    sns.barplot(x="prompt", y="constraint satisfaction", data=pd.DataFrame(df), ax=axes[0])
    sns.barplot(x="prompt", y="TVD", data=pd.DataFrame(df), ax=axes[1])
    sns.barplot(x="prompt", y="TVD-NF", data=pd.DataFrame(df), ax=axes[2])
    fig.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
