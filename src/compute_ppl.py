# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import argparse
import yaml
import numpy as np
from discontrol.scorer import SingleWordFeature, GenderFeature, MultiWordFeature, ExponentialScorer
from discontrol.pipeline import build_distribution, build_ebm_from_config
from discontrol.distribution.samplerepo import SampleRepository
from discontrol.misc import list_array
from pathlib import Path
import json
from transformers import AutoTokenizer

def get_model_name(experiment):
    if experiment == "female_science":
        return "mkhalifa/gpt2-biographies"
    elif experiment == "amazing":
        return "gpt2"
    elif experiment == 'wikileaks':
        return "gpt2"

def get_base(dist_cfg):
    with open(dist_cfg, 'r') as fdistribution_config:
        distribution_config = yaml.safe_load(fdistribution_config)
    a = build_distribution(distribution_config['base'])
    print(a.model.name_or_path)
    return a

def main(args):
    sample_file = Path(args.sample_file)
    samples = np.load(sample_file, allow_pickle=True)
    print(f"Loaded {len(samples)} samples from disk")
    scores = np.load(sample_file.with_suffix(".base.pkl"), allow_pickle=True)
    print(f"Loaded {len(scores)} scores from disk")
    idx = 0
    
    model_name = get_model_name(args.experiment)
    print("Loaded model")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print(tokenizer)
    print("Loaded tokenizer")

    ce = 0
    length = 0
    for idx, sample in enumerate(samples):
        loga = scores[idx]
        length += len(tokenizer.encode(sample))
        ce += -loga

    ppl = np.exp(ce/length)
    results = {}
    results['sample_file'] = args.sample_file
    results['total_samples'] = len(samples)
    results['sample_size'] = args.sample_size
    results['total_scores'] = len(scores)
    results['ppl'] = ppl
    with open(Path(args.sample_file).with_suffix('.ppl.json'), 'w') as fout:
        json.dump(results, fout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_file", type=str)
    parser.add_argument("--sample_size", type=int)
    parser.add_argument("--experiment", type=str)
    args = parser.parse_args()
    main(args)
