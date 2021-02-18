#!/usr/bin/env python3
# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import os, sys
import json
import argparse
import numpy as np
import importlib
from pathlib import Path
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import json
from discontrol.pipeline import build_sampler_from_config
from discontrol.misc import list_array
import pickle
from tqdm import *
import logging
logging.basicConfig(level=logging.INFO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--samples', type=int, default=10000, help='total number of samples')
    ap.add_argument('-o', '--output', type=Path, help='output filename')
    ap.add_argument('-ec', '--ebm-config', help='target EBM configuration file')
    ap.add_argument('-dc', '--distribution-config', help='proposal distribution configuration file')
    ap.add_argument('-sc', '--sampler-config', help='sampler configuration file')
    ap.add_argument('-pc', '--proposal-config', help='proposal distribution configuration file')
    ap.add_argument('-db', '--proposal-batch', type=int, default=100, help='proposal batch size')
    ap.add_argument('-sb', '--sampler-batch', type=int, default=100, help='sampler batch size')
    args = ap.parse_args()
    config = load_config(args)
    sys.stdout.write('Loading... ')
    sys.stdout.flush()
    sampler = build_sampler_from_config(config, cuda=True)
    sys.stdout.write('[done]\n')
    sys.stdout.flush()
    total_samples = 0
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fout = open(args.output, 'w')
    else:
        fout = sys.stdout
    all_samples = None
    with fout, tqdm(total=args.samples) as progress:
        while total_samples < args.samples:
            samples = sampler.sample()
            total_samples += len(samples)
            progress.update(len(samples))
            write(samples, fout)
            all_samples = samples.concat_to(all_samples)
    if args.output:
        meta_fn = args.output.with_suffix('.meta.json')
        pkl_fn = args.output.with_suffix('.pkl')
        pickle_samples(pkl_fn, all_samples)
        save_meta(meta_fn, args, sampler)

def load_config(args):
    default_config = {'sampler': {'class': 'DirectSampler'}}
    config = default_config
    for config_fn in [args.distribution_config, args.ebm_config, args.sampler_config, args.proposal_config]:
        if config_fn:
            config.update(yaml.load(open(config_fn), Loader=Loader))
    if 'base' not in config:
        logging.warning("Target EBM not set")
        config['base'] = config['proposal']
    config['base']['args']['num_return_sequences'] = args.proposal_batch
    if 'proposal' in config:
        config['proposal']['args']['num_return_sequences'] = args.proposal_batch
    if 'sampler' in config:
        if not 'args' in config['sampler']:
            config['sampler']['args'] = {}
        config['sampler']['args']['batch_size'] = args.sampler_batch
    return config

def write(samples, fout):
    for sample in samples:
        fout.write(json.dumps(sample) + '\n')

def save_meta(meta_fn, args, sampler):
        meta = {}
        meta['args'] = vars(args)
        meta['args']['distribution_config'] = os.path.abspath(args.distribution_config) if args.distribution_config else None
        if args.ebm_config:
            meta['args']['ebm_config'] = os.path.abspath(args.ebm_config)
        meta['args']['sampler_config'] = os.path.abspath(args.sampler_config) if args.sampler_config else None
        if args.proposal_config:
            meta['args']['proposal_config'] = os.path.abspath(args.proposal_config)
        del(meta['args']["output"])
        if 'SLURM_JOB_ID' in os.environ:
            meta['args']['SLURM_JOB_ID'] = os.environ['SLURM_JOB_ID']
        meta['sampler'] = sampler.get_meta()
        meta['metrics'] = {}
        try:
            meta['metrics']['acceptance_rate'] = sampler.get_acceptance_rate()
        except:
            pass
        with open(meta_fn, 'w') as fmeta:
            json.dump(meta, fmeta)

def pickle_samples(pkl_fn, all_samples):
    with open(pkl_fn, 'wb') as fout:
        pickle.dump(all_samples, fout)

if __name__ == '__main__':
    main()
