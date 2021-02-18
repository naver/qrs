# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import multiprocessing
import argparse

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
from discontrol.scorer import SingleWordFeature, GenderFeature, MultiWordFeature
from discontrol.pipeline import build_sampler_from_config
from discontrol.metrics import Distinct_N, SelfBLEU
from discontrol.misc import NumpyArray

from qrs_tvd import  get_tvd_estimates, add_bounds, project_into_curve_x

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

def main(samples_dir, experiment):

    #experiment = st.selectbox("experiment", )
    sample_size = 50_000

    # Load data
    P, q, a = load_samples(experiment)
    constraints = get_constraints(experiment)
    print(f"* Loaded {P.size:,} samples from disk.")
    ### Collect samples from QRS.
    print("## Running the QRS sampler")

    # QRS
    q.batch_size = sample_size
    acceptance_rates = [1e-1, 1e-2, 1e-3, 1e-4]
    for ar in acceptance_rates:
        samples_file = samples_dir / f"QRS-{ar:.6f}-{sample_size}-{experiment}.npy"
        if samples_file.exists():
            s = np.load(samples_file, allow_pickle=True)
            print(f"* Loaded {sample_size} samples for QRS@{ar}")
        else:
            qrs = QuasiRejectionSampler(P, q, sample_size, min_acceptance_rate=ar)
            s = qrs.sample()
            np.save(samples_file, s, allow_pickle=True)
            print(f"* Sampled {sample_size} times from QRS@{ar}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=["female_science", "female_sports", "amazing", "female"])
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = Path(base_dir) / "samples"
    samples_dir.mkdir(exist_ok=True)
    main(samples_dir, args.experiment)
