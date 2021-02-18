# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse

from cycler import cycler

from transformers import MarianMTModel, MarianTokenizer

from discontrol.scorer import KeywordsFeature, ParaphraseFeature, ProductScorer
from discontrol.distribution.nmt import FBRoundTripNMT
from discontrol.distribution.lm import HFAutoregressiveLM
from discontrol.sampler import QuasiRejectionSamplerFixedBeta
from discontrol.distribution.samplerepo import SampleRepository

from pathlib import Path

def load_samples(seq_idx):
    base_dir = Path(f"<output-dir>/paraphrasing-95/{seq_idx}/")
    samples_file = base_dir / "round-trip-nmt.npy"
    q = SampleRepository(samples_file, base_dir / "round-trip-nmt.proposal.npy")
    P_sup = SampleRepository(samples_file, base_dir / "round-trip-nmt.ebm-sup.npy")
    P_unsup = SampleRepository(samples_file, base_dir / "round-trip-nmt.ebm-unsup.npy")
    feature_sup = SampleRepository(samples_file, base_dir / "round-trip-nmt.feature-sup.npy")
    feature_unsup = np.load(base_dir / "round-trip-nmt.feature-unsup.npy")
    return q, feature_unsup, P_unsup, feature_sup, P_sup

def main(seq_idx):

    orig = [
        'How is the two wheeler insurance from Bharti Axa insurance?',
        'Are there Doctor Who references in the Muse song "Knights of Cydonia"?',
        'In French, how do you say "cool"?'
    ]

    fixed_betas = [
        2.120950887920198e-24,
        #2.2229964825262004e-23,
        #7.196856730011513e-8
        7.196856730011536e-23,
        5.179474679231219e-8
    ]

    input_seq = orig[seq_idx]
    q, feature_unsup, P_unsup, feature_sup, P_sup = load_samples(seq_idx)
    print(f"* loaded {len(feature_unsup)} samples from disk")

    #### q samples
    print("==== q")
    q.batch_size = 12
    q_samples = q.sample()
    for s in q_samples:
        print(s)

    ### QRS samples
    print("==== QRS")
    beta = fixed_betas[seq_idx]
    q.reset()
    q.batch_size = 1
    qrs = QuasiRejectionSamplerFixedBeta(P_sup, q, beta=beta, batch_size=1, silent=True)
    try:
        for _ in range(12):
            qrs_samples = qrs.sample()
            for s in qrs_samples:
                print(s)
    except: pass
    print(qrs.accepted)
    print(qrs.total)
    print(qrs.get_acceptance_rate())
    print(q.size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_idx', type=int)
    args = parser.parse_args()
    main(args.seq_idx)
