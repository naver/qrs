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

from pathlib import Path

from discontrol.sampler import QuasiRejectionSampler, IMHSampler
from discontrol.distribution.poisson import PoissonDistribution
from discontrol.scorer import SingleWordFeature, GenderFeature, MultiWordFeature

from gdc_vs_qrs import get_constraints

# Plot settings
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
         '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

def main():
    sample_file = "<output-dir>/female_science/base-female_science.pkl"
    samples = np.load(sample_file, allow_pickle=True)
    print(f"* loaded {len(samples):,} samples.")

    print("### Female Science")
    constraints = get_constraints("female_science")
    scorer_female = constraints[0][-1]
    scorer_science = constraints[1][-1]

    female = scorer_female.score(samples).data
    science = scorer_science.score(samples).data
    only_female = np.logical_and(female, np.logical_not(science))
    only_science = np.logical_and(science, np.logical_not(female))
    both = np.logical_and(female, science)
    none = np.logical_not(female+science+both)
    print(only_female.sum())
    print(only_science.sum())
    print(both.sum())
    print(none.sum())
    print(f"lambda_fsc = {np.log(only_science.sum() / both.sum())}")

    print("### Female Sports")
    constraints = get_constraints("female_sports")
    scorer_female = constraints[0][-1]
    scorer_sports = constraints[1][-1]

    sports = scorer_sports.score(samples).data
    only_female = np.logical_and(female, np.logical_not(sports))
    only_sports = np.logical_and(sports, np.logical_not(female))
    both = np.logical_and(female, sports)
    none = np.logical_not(female+sports+both)
    print(only_female.sum())
    print(only_sports.sum())
    print(both.sum())
    print(none.sum())
    print(f"lambda_fsp = {np.log(only_sports.sum() / both.sum())}")

    print("### Female")
    none = np.logical_not(female)
    print(female.sum())
    print(none.sum())
    print(f"lambda_f = {np.log(none.sum() / female.sum())}")

if __name__ == "__main__":
    main()
