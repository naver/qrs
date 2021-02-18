#!/usr/bin/env python
# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import os, sys
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, ".."))
import argparse
from pathlib import Path
import json
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import yaml
from pipeline import build_classes_and_args
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm
from fast_bleu import SelfBLEU
import nltk
import numpy as np
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt2')
    ap.add_argument('filename', type=Path)
    args = ap.parse_args()

    sentences = [json.loads(l) for l in open(args.filename)]
    meta = json.load(open(args.filename.with_suffix('.meta.json')))
    features = []
    if 'distribution_config' in meta or 'distribution_config' in meta['args']:
        try:
            config_fn = meta['args']['distribution_config']
        except:
            config_fn = meta['distribution_config']
        config = yaml.load(open(config_fn), Loader=Loader)
        features = build_classes_and_args('scorer', config['filters'])
        try:
            features = features[0].features # exponential scorer
        except:
            pass
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    metrics = evaluate(model, tokenizer, features, sentences)
    metrics.update(self_bleu(sentences))
    with open(args.filename.with_suffix('.eval.json'), 'w') as fout:
        json.dump(metrics, fout)


def evaluate(model, tokenizer, features, sentences):
    ppl = 0
    satisfaction = {feature.name: 0 for feature in features}
    n = 0
    for s in tqdm.tqdm(sentences):
        s_ids = tokenizer.encode(s, return_tensors='pt')
        ppl += model(s_ids, labels=s_ids, return_dict=True).loss.exp().item()
        n += 1
        for feature in features:
            satisfaction[feature.name] += feature.score([s])[0]
    res = {}
    for feature_name, score in satisfaction.items():
        res[feature_name] = score / n
    res['ppl'] = ppl / n
    return res

def self_bleu(sentences):
    res = {}
    tokenized_sentences = [nltk.word_tokenize(s) for s in sentences]
    bleu = SelfBLEU(tokenized_sentences,
            {'BLEU-2': (1/2., 1/2.), 'BLEU-3': (1/3., 1/3., 1/3.)})
    scores = bleu.get_score()
    return {k: np.mean(v) for k,v, in scores.items()}

def strip_eos(sentences):
    return [s[len('<|endoftext|>'):] if '<|endoftext|>' in s else s for s in sentences]

main()
