# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import argparse
import numpy as np
from pathlib import Path
import json

from discontrol.metrics import Distinct_N, SelfBLEU

def main(args):
    samples = np.load(args.sample_file, allow_pickle=True)
    selfbleu = SelfBLEU(gram=5)
    dist1 = Distinct_N(1)
    dist2 = Distinct_N(2)
    dist3 = Distinct_N(3)
    idx = 0
    selfbleus = []
    dist1s = []
    dist2s = []
    dist3s = []
    lengths = []
    p_uniqs = []
    while idx < len(samples):
        S = samples[idx:idx+args.sample_size]
        selfbleus.append(selfbleu.compute_metric(S)) 
        dist1s.append(dist1.compute_metric(S)) 
        dist2s.append(dist2.compute_metric(S)) 
        dist3s.append(dist3.compute_metric(S)) 
        p_uniqs.append(len(np.unique(S)) / len(S))
        lengths.append(len(S))
        idx += args.sample_size

    results = {}
    results['sample_file'] = args.sample_file
    results['total_samples'] = len(samples)
    results['lengths'] = lengths
    results['selfbleus'] = selfbleus
    results['dist1s'] = dist1s
    results['dist2s'] = dist2s
    results['dist3s'] = dist3s
    results['mean_dist1s'] = np.mean(dist1s)
    results['std_dist1s'] = np.std(dist1s)
    results['mean_dist2s'] = np.mean(dist2s)
    results['std_dist2s'] = np.std(dist2s)
    results['mean_dist3s'] = np.mean(dist3s)
    results['std_dist3s'] = np.std(dist3s)
    results['mean_selfbleus'] = np.mean(selfbleus)
    results['std_selfbleus'] = np.std(selfbleus)
    results['p_uniqs'] = p_uniqs
    results['mean_p_uniqs'] = np.mean(p_uniqs)
    results['std_p_uniqs'] = np.std(p_uniqs)
    with open(Path(args.sample_file).with_suffix('.selfbleu.json'), 'w') as fout:
        json.dump(results, fout)

    print(args.sample_file)
    print(args.sample_size, len(samples))
    print(lengths)
    print("== self-BLEU")
    print(selfbleus)
    print(np.mean(selfbleus))
    print(np.std(selfbleus))
    print("== p uniq")
    print(p_uniqs)
    print(np.mean(p_uniqs))
    print(np.std(p_uniqs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_file", type=str)
    parser.add_argument("--sample_size", type=int)
    args = parser.parse_args()
    main(args)
