# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import argparse
import numpy as np
from discontrol.scorer import SingleWordFeature, GenderFeature, MultiWordFeature, ExponentialScorer
import json
from pathlib import Path

def get_constraints(experiment):
    if experiment == "amazing":
        return [("amazing", 1., SingleWordFeature("amazing"))]
    elif experiment == "wikileaks":
        return [("wikileaks", 1., SingleWordFeature("Wikileaks"))]
    elif experiment == "female":
        return [("female", 0.5, GenderFeature("female"))]
    elif experiment == "female_science":
        return [("female", 0.5, GenderFeature("female")),
                ("science", 1., MultiWordFeature("resources/wikibio-wordlists/science.txt"))]
    elif experiment == "female_sports":
        return [("female", 0.5, GenderFeature("female")),
                ("sports", 1., MultiWordFeature("resources/wikibio-wordlists/sports.txt"))]

def main(args):
    samples = np.load(args.sample_file, allow_pickle=True)
    constraints = get_constraints(args.experiment)
    idx = 0
    moments = {constraint[0]: [] for constraint in constraints}
    lengths = []
    while idx < len(samples):
        S = samples[idx:idx+args.sample_size]
        lengths.append(len(S))
        for constraint in constraints:
            name, _, scorer = constraint
            score = np.average(scorer.score(S))
            moments[name].append(score)
        idx += args.sample_size

    results = {}
    results['sample_file'] = args.sample_file
    results['sample_size'] = len(samples)
    results['lengths'] = lengths
    results['moments'] = {}

    print(args.sample_file)
    print(args.sample_size, len(samples))
    print(lengths)
    print("== MOMENTS")
    for constraint in constraints:
        name, target, _ = constraint
        print(f"constraint: {target:.0%} {name}")
        results['moments'][name] = {}
        results['moments'][name]['scores'] = moments[name]
        results['moments'][name]['mean'] = np.mean(moments[name])
        results['moments'][name]['std'] = np.std(moments[name])
        print(moments[name])
        print(results['moments'][name]['mean'])
        print(results['moments'][name]['std'])
    print("==")
    with open(Path(args.sample_file).with_suffix('.moments.json'), 'w') as fout:
        json.dump(results, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_file", type=str)
    parser.add_argument("--sample_size", type=int)
    parser.add_argument("--experiment", type=str)
    args = parser.parse_args()
    main(args)
