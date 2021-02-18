#!/usr/bin/env python3
# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import argparse
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from pathlib import Path
import numpy as np
from discontrol.pipeline import build_ebm_from_config, build_distribution
from discontrol.sampler import AccumulatorSampler

# expected number of accepted samples to use in the estimation of the AR
K = 100

# number of sequences returned by the proposal in parallel
proposal_batch_size = 100

# number of concurrent samples used in estimating
scoring_batch_size = 100

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output-config', type=Path, help='output filename')
    ap.add_argument('-ec', '--ebm-config', help='target EBM configuration file')
    ap.add_argument('-pc', '--proposal-config', help='proposal distribution configuration file')
    ap.add_argument('-ar', '--target-ar', type=float, required=True)
    args = ap.parse_args()

    config = load_config(args)
    ebm = build_ebm_from_config(config)
    proposal = build_distribution(config['proposal']).to('cuda')
    beta = estimate_beta(ebm, proposal, args.target_ar)
    sampler_config = {'sampler': 
            {'class': 'RejectionSampler', 'args': {'beta': beta, 'truncate': True}}}
    with open(args.output_config, 'w') as fout:
        fout.write(yaml.dump(sampler_config, Dumper=Dumper))


def load_config(args):
    config = {}
    for config_fn in [args.ebm_config, args.proposal_config]:
        if config_fn:
            config.update(yaml.load(open(config_fn), Loader=Loader))
    config['proposal']['args']['num_return_sequences'] = proposal_batch_size
    return config

def estimate_beta(ebm, proposal, target_ar):
    N = int(1/target_ar) * K
    x = AccumulatorSampler(proposal, N).sample()
    log_q_x = proposal.log_score_batched(x, scoring_batch_size)
    log_P_x = ebm.log_score_batched(x, scoring_batch_size)
    beta_x = (log_P_x - log_q_x).exp()
    sorted_beta_x = np.sort(beta_x)
    b_x = np.cumsum(sorted_beta_x) /sorted_beta_x
    a_x = np.arange(len(x), 0, -1) - 1
    ar_at_beta_x = (a_x + b_x) / len(x)
    idx_at_ar = len(x) - 1 - np.searchsorted(np.flip(ar_at_beta_x), target_ar, side='right') 
    beta_at_ar = sorted_beta_x[idx_at_ar-1] # picks the beta with AT LEAST the given acceptance rate
    return float(beta_at_ar)

main()
