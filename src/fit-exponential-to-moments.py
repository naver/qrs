#!/usr/bin/env python3
# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import os, sys

import argparse
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from pathlib import Path
from discontrol.pipeline import build_class_and_args
from discontrol.scorer import ExponentialScorer

from pathlib import Path
from discontrol.distribution.samplerepo import SampleRepository

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--moments_config', type=Path, required=True)
    ap.add_argument('-b', '--batch', type=int, default=512, help='batch size')
    ap.add_argument('--nsamples', type=int, default=4096, help='number of samples used')
    ap.add_argument('--max-iter', type=int, default=100, help='maximum number of training iterations')
    ap.add_argument('--tolerance', type=float, default=1e-2, help='early stopping loss tolerance')
    ap.add_argument("--lr", type=float, default=5, help="learning rate")
    args = ap.parse_args()

    config = yaml.load(open(args.moments_config), Loader=Loader)
    config['base']['args']['num_return_sequences'] = args.batch
    base_lm, proposal, features, moments = load_config(config)
    del(config['base']['args']['num_return_sequences'])
    exp = ExponentialScorer.fit(base_lm, features, moments, tolerance=args.tolerance, nsamples=args.nsamples,
                                iterations=args.max_iter, lr=args.lr, proposal_distr=proposal, batch_size=args.batch)
    distr_config = build_distr_config(config, exp.coefficients)
    print(distr_config)
    print('Target moments:', moments)
    print('Estimated moments:', exp.estimate_moments(base_lm, nsamples=args.nsamples))
    print('Estimated moments (using proposal):', exp.estimate_moments(base_lm, nsamples=args.nsamples, proposal_distr=proposal))

def load_config(config):
    base_lm = build_class_and_args('distribution', config['base'])#.to('cuda')
    proposal = None
    if "proposal" in config:
        proposal = build_class_and_args('distribution', config['proposal'])#.to('cuda')
    features = []
    moments = []
    for feature_moments in config['moments']:
        feature = build_class_and_args('scorer', feature_moments['feature'])
        moment = feature_moments['moment']
        features.append(feature)
        moments.append(moment)
    return base_lm, proposal, features, moments

def build_distr_config(config, coefficients):
    distr_config = dict(config)
    del(distr_config['moments'])
    distr_config['filters'] = [{
        'class': 'ExponentialScorer', 
        'args': { 
            'features': [m['feature'] for m in config['moments']],
            'coefficients': coefficients.data.cpu().numpy().tolist()}}]
    return distr_config

if __name__ == '__main__':
    main()
