#!/usr/bin/env python
# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import os, sys
base_dir = os.path.dirname(os.path.abspath(__file__))
import argparse
from pathlib import Path
import json
import tqdm
import yaml
import numpy as np
import pickle
from discontrol.pipeline import build_distribution, build_ebm_from_config, build_class_and_args
from discontrol.scorer import BaseScorer
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=Path)
    ap.add_argument('-b', '--batch-size', type=int, default=50)
    ap.add_argument('-s', '--scorer', choices=['proposal', 'ebm', 'base', 'features'], required=True)
    ap.add_argument('--reset-tokenization', action='store_true', default=False)
    args = ap.parse_args()

    meta = load_meta(args)
    samples = load_samples(args)
    scorer = get_scorer(meta, args.scorer)
    if args.scorer != 'features':
        all_scores = score_distr(scorer, samples, args)
    else:
        all_scores = score_features(scorer, samples, args)
    save_scores(all_scores, args)

def load_samples(args):
    with open(args.samples.with_suffix('.pkl'), 'rb') as fsamples:
        samples = pickle.load(fsamples)
    if args.reset_tokenization:
        samples.aux_data = {} # this avoids some bugs due to the wrong tokenization being loaded
    return samples

def load_meta(args):
    with open(args.samples.with_suffix('.meta.json'), 'r') as fmeta:
        meta = json.load(fmeta)
    return meta

def score_distr(scorer, samples, args):
    bsz = args.batch_size
    all_scores = []
    for i in tqdm.trange(0, len(samples), bsz):
        batch = samples[i:i+bsz]
        scores = scorer.log_score(batch).numpy()
        all_scores.append(scores)
    all_scores = np.concatenate(all_scores)
    return all_scores

def score_features(scorers, samples, args):
    scores = []
    for x in tqdm.tqdm(samples):
        scores_x = {scorer_name: scorer.score([x])[0]
                for scorer_name, scorer in scorers.items()}
        scores.append(scores_x)
    return pd.DataFrame(scores)

def save_scores(all_scores, args):
    with open(args.samples.with_suffix(f'.{args.scorer}.pkl'), 'wb') as fout:
        pickle.dump(all_scores, fout)

def get_scorer(meta, scorer_name):
    if scorer_name == 'proposal':
        return get_proposal(meta).to('cuda')
    elif scorer_name == 'base':
        return get_base(meta).to('cuda')
    elif scorer_name == 'ebm':
        return get_ebm(meta).to('cuda')
    elif scorer_name == 'features':
        return get_features_scorer(meta)

def get_proposal(meta):
    try:
        with open(meta['args']['sampler_config'], 'r') as fsampler_config:
            sampler_config = yaml.safe_load(fsampler_config)
        #NB: Hacky! (inject prompt_acceptance_rate info)
        if 'prompt_acceptance_rate' in meta['sampler']:
            sampler_config['proposal']['args']['_prompt_acceptance_rate'] = meta['sampler']['prompt_acceptance_rate']
        proposal = build_distribution(sampler_config['proposal'])
    except KeyError:
        with open(meta['args']['proposal_config'], 'r') as fproposal_config:
            proposal_config = yaml.safe_load(fproposal_config)
        proposal = build_distribution(proposal_config['proposal'])

    return proposal

def get_ebm(meta):
    args = meta['args']
    dist_cfg = args['distribution_config'] if args['distribution_config'] else args['ebm_config']
    with open(dist_cfg, 'r') as fdistribution_config:
        distribution_config = yaml.safe_load(fdistribution_config)
    P = build_ebm_from_config(distribution_config)
    return P

def get_base(meta):
    args = meta['args']
    dist_cfg = args['distribution_config'] if args['distribution_config'] else args['ebm_config']
    with open(dist_cfg, 'r') as fdistribution_config:
        distribution_config = yaml.safe_load(fdistribution_config)
    a = build_distribution(distribution_config['base'])
    return a

def get_features_scorer(meta):
    args = meta['args']
    dist_cfg = args['distribution_config'] if args['distribution_config'] else args['ebm_config']
    with open(dist_cfg, 'r') as fdistribution_config:
        distribution_config = yaml.safe_load(fdistribution_config)
    scorers = get_scorers(distribution_config['filters'])
    return scorers

def get_scorers(filters):
    scorers = {}
    for fltr in filters:
        if fltr['class'] == 'ExponentialScorer':
            scorers.update(get_scorers(fltr['args']['features']))
        else:
            scorer_name, scorer = build_scorer(fltr)
            scorers[scorer_name] = scorer
    return scorers

def build_scorer(cfg):
    if cfg['class'] == 'SingleWordFeature':
        name = cfg['args']['word']
    elif cfg['class'] == 'MultiWordFeature':
        name = Path(cfg['args'][0]).stem
    elif cfg['class'] == 'GenderFeature':
        name = cfg['args'][0]
    else:
        raise RuntimeError(f"Unknown scorer class {cfg['class']}")
    return name, build_class_and_args('scorer', cfg)

class DictionaryScorer(BaseScorer):
    def __init__(self, scorers):
        super(DictionaryScorer, self).__init__()
        self.scorers = scorers

    def score(self, x):
        return {scorer_name: scorer.score(x)
                for scorer_name, scorer in self.scorers.items()}

if __name__ == '__main__':
    main()
