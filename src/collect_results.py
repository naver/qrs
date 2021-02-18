#!/usr/bin/env python
# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import streamlit as st
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from pprint import pprint
from itertools import *
import pandas as pd
from pandas.api.types import is_numeric_dtype

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', type=Path, default='<output-dir>/10k/')
    args = ap.parse_args()

    moments = collect_moments(args)
    selfbleus, uniqs, dist1s, dist2s, dist3s = collect_diversity(args)
    ppls = collect_ppls(args)
    ars = collect_effective_ars(args)
    results = agg_results(moments, selfbleus, uniqs, dist1s, dist2s, dist3s, ppls, ars)
    df = pd.DataFrame(results).transpose()
    df = df.drop('wikileaks', level=0)
    df = df.drop('qrs-0.1', level=1)
    df = df.drop('qrs-0.001', level=1)
    df = simplify_columns(df)
    st.write(df)

    df.loc[pd.IndexSlice[:,'direct'], 'tgt_ar'] = '$1$'
    df.loc[pd.IndexSlice[:,'qrs-amazing-0.1'], 'tgt_ar'] = '$10^{-1}$'
    df.loc[pd.IndexSlice[:,'qrs-amazing-0.001'], 'tgt_ar'] = '$10^{-3}$'
    df.loc[pd.IndexSlice[:,'qrs-female_science-0.1'], 'tgt_ar'] = '$10^{-1}$'
    df.loc[pd.IndexSlice[:,'qrs-female_science-0.001'], 'tgt_ar'] = '$10^{-3}$'
    df.loc[pd.IndexSlice[:,'imh-0.1'], 'tgt_ar'] = '$10^{-1}$'
    df.loc[pd.IndexSlice[:,'imh-0.001'], 'tgt_ar'] = '$10^{-3}$'
    df.loc[pd.IndexSlice[:,'imhreset-0.1'], 'tgt_ar'] = '$10^{-1}$'
    df.loc[pd.IndexSlice[:,'imhreset-0.001'], 'tgt_ar'] = '$10^{-3}$'
    df.loc[pd.IndexSlice[:,'rwmh-0.1'], 'tgt_ar'] = '$10^{-1}$'
    df.loc[pd.IndexSlice[:,'rwmhreset-0.1'], 'tgt_ar'] = '$10^{-1}$'
    df.loc[pd.IndexSlice[:,'rwmh-0.001-amazing-0.1'], 'tgt_ar'] = '$10^{-3}$'
    df.loc[pd.IndexSlice[:,'rwmh-0.001-amazing-0.01'], 'tgt_ar'] = '$10^{-3}$'
    df.loc[pd.IndexSlice[:,'rwmhreset-0.001-amazing-0.1'], 'tgt_ar'] = '$10^{-3}$'
    df.loc[pd.IndexSlice[:,'rwmhreset-0.001-amazing-0.01'], 'tgt_ar'] = '$10^{-3}$'
    df.loc[pd.IndexSlice[:,'imh-0.1'], 'ppl'] = '-'
    df.loc[pd.IndexSlice[:,'imh-0.001'], 'ppl'] = '-'
    df.loc[pd.IndexSlice[:,'rwmh-0.1'], 'ppl'] = '-'
    df.loc[pd.IndexSlice[:,'rwmh-0.001-amazing-0.1'], 'ppl'] = '-'
    df.loc[pd.IndexSlice[:,'rwmh-0.001-amazing-0.01'], 'ppl'] = '-'
    df['tvd'] = '{\color[HTML]{980000} Unk.}'
    df['kl'] = '{\color[HTML]{980000} Unk.}'
    df.loc[pd.IndexSlice['amazing', 'qrs-amazing-0.1'], 'tvd'] = '$0.27 \pm 0.0054$'
    df.loc[pd.IndexSlice['amazing', 'qrs-amazing-0.1'], 'kl'] = '$0.45 \pm 0.045$'
    df.loc[pd.IndexSlice['amazing', 'qrs-amazing-0.001'], 'tvd'] = '$0.01 \pm 0.0067$'
    df.loc[pd.IndexSlice['amazing', 'qrs-amazing-0.001'], 'kl'] = '$0.011 \pm 0.0093$'
    df.loc[pd.IndexSlice['amazing', 'direct'], 'tvd'] = '$0.67 \pm 0.00095$'
    df.loc[pd.IndexSlice['amazing', 'direct'], 'kl'] = '$1.91 \pm 0.04$'
    df.loc[pd.IndexSlice['female_science', 'qrs-female_science-0.1'], 'tvd'] = '$0.37 \pm 0.0049$'
    df.loc[pd.IndexSlice['female_science', 'qrs-female_science-0.1'], 'kl'] = '$0.77 \pm 0.036$'
    df.loc[pd.IndexSlice['female_science', 'qrs-female_science-0.001'], 'tvd'] = '$0.012 \pm 0.0055$'
    df.loc[pd.IndexSlice['female_science', 'qrs-female_science-0.001'], 'kl'] = '$0.0084 \pm 0.0048$'
    df.loc[pd.IndexSlice['female_science', 'direct'], 'tvd'] = '$0.7 \pm 0.00099$'
    df.loc[pd.IndexSlice['female_science', 'direct'], 'kl'] = '$2.28 \pm 0.03$'
    st.write(df.sort_index())
    df = df.drop('rwmhreset-0.001-amazing-0.01', level=1)
    df = df.drop('rwmh-0.001-amazing-0.01', level=1)

    #df[df['sampler']=='qrs-0.001', 'tgt_ar'] = '$10^{-1}$'
    df = df.reset_index().rename(columns={'level_0': 'ebm', 'level_1': 'sampler'}).sort_values(['tgt_ar', 'sampler'])
    st.write(df)
    df['sampler'] = df['sampler'].str.replace('direct', 'DPG')
    df['sampler'] = df['sampler'].str.replace('qrs.*', 'QRS')
    df['sampler'] = df['sampler'].str.replace('imhreset.*', 'IMH-R')
    df['sampler'] = df['sampler'].str.replace('imh.*', 'IMH')
    df['sampler'] = df['sampler'].str.replace('rwmhreset-0.1', 'RWMH-R-base')
    df['sampler'] = df['sampler'].str.replace('rwmhreset-0.001-amazing-0.1', 'RWMH-R')
    df['sampler'] = df['sampler'].str.replace('rwmhreset-0.001-amazing-0.01', 'RWMH-R')
    df['sampler'] = df['sampler'].str.replace('rwmh-0.001-amazing-0.1', 'RWMH')
    df['sampler'] = df['sampler'].str.replace('rwmh-0.001-amazing-0.01', 'RWMH')
    df['sampler'] = df['sampler'].str.replace('rwmh-0.1', 'RWMH-base')

    df1 = df[df['ebm']=='amazing'][['sampler', 'tgt_ar', 'amazing', 'ppl', 'selfbleu', 'uniqs', 'dist1', 'dist2', 'dist3', 'tvd', 'kl']]
    names = {"kl": r"$\KL(p, p_\beta)\downarrow$",
        "tvd": r"$\TVD(p, p_\beta)\downarrow$",
        "tgt_ar": "AR",
        "dist1": r'Dist-1$\uparrow$',
        "dist2": r'Dist-2$\uparrow$',
        "dist3": r'Dist-3$\uparrow$',
        "ppl": r'PPL$\downarrow$',
        "uniqs": r'\%Uniq$\uparrow$',
        'selfbleu': r'Self-BLEU-5$\downarrow$',
        'amazing': r'\%amazing',
        'female': r'\%female',
        'science': r'\%science'
        }
    df1.rename(columns=names, inplace=True)
    df2 = df[df['ebm']=='female_science'][['sampler', 'tgt_ar', 'female', 'science', 'ppl', 'selfbleu', 'uniqs', 'dist1', 'dist2', 'dist3', 'tvd', 'kl']]
    df2.rename(columns=names, inplace=True)
    st.code(df1.to_latex(escape=False, index=False))
    st.code(df2.to_latex(escape=False, index=False))

def collect_moments(args):
    moments = defaultdict(lambda:defaultdict(list))
    for fn in args.path.glob('*/*/*.moments.json'):
        sampler = fn.stem[:-len(".moments")]
        experiment = fn.parent.name
        moments_data = json.load(open(fn))
        for constraint_name, constraint_data in moments_data['moments'].items():
            moments[(experiment, sampler)][constraint_name].append(constraint_data['mean']*100)
    return moments

def collect_diversity(args):
    selfbleus = defaultdict(list)
    uniqs = defaultdict(list)
    dist1s = defaultdict(list)
    dist2s = defaultdict(list)
    dist3s = defaultdict(list)
    for fn in args.path.glob('*/*/*.selfbleu.json'):
        sampler = fn.stem[:-len(".selfbleu")]
        experiment = fn.parent.name
        selfbleu_data = json.load(open(fn))
        uniqs[(experiment, sampler)].append(selfbleu_data['mean_p_uniqs']*100)
        selfbleus[(experiment, sampler)].append(selfbleu_data['mean_selfbleus']*100)
        try:
            dist1s[(experiment, sampler)].append(selfbleu_data['mean_dist1s']*100)
            dist2s[(experiment, sampler)].append(selfbleu_data['mean_dist2s']*100)
            dist3s[(experiment, sampler)].append(selfbleu_data['mean_dist3s']*100)
        except KeyError:
            print('(diversity) Error loading', fn)
    return selfbleus, uniqs, dist1s, dist2s, dist3s

def collect_ppls(args):
    ppls = defaultdict(list)
    for fn in args.path.glob('*/*/*.ppl.json'):
        sampler = fn.stem[:-len(".ppl")]
        experiment = fn.parent.name
        data = json.load(open(fn))
        try:
            ppls[(experiment, sampler)].append(data['ppl'])
        except KeyError:
            print('ERROR loading', fn)
    return ppls

def collect_effective_ars(args):
    ars = defaultdict(list)
    for fn in args.path.glob('*/*/*.meta.json'):
        sampler = fn.stem[:-len(".meta")]
        experiment = fn.parent.name
        data = json.load(open(fn))
        if 'qrs' not in sampler:
            continue
        if 'metrics' in data and 'acceptance_rate' in data['metrics'] and data['metrics']['acceptance_rate']:
            ars[(experiment, sampler)].append(data['metrics']['acceptance_rate'])
    return ars


def agg_results(moments, selfbleus, uniqs, dist1s, dist2s, dist3s, ppls, ars):
    results = defaultdict(dict)
    for key, experiment_moments in moments.items():
        for constraint_name, constraint_moments in experiment_moments.items():
            results[key][f'{constraint_name}_mean'] = np.mean(constraint_moments)
            results[key][f'{constraint_name}_std'] = np.std(constraint_moments)
    add_metric(results, selfbleus, 'selfbleu')
    add_metric(results, uniqs, 'uniqs')
    add_metric(results, dist1s, 'dist1')
    add_metric(results, dist2s, 'dist2')
    add_metric(results, dist3s, 'dist3')
    add_metric(results, ppls, 'ppl')
    add_metric(results, ars, 'ar')
    return results

def add_metric(results, metric_dict, metric_name):
    for key, metric_vals in metric_dict.items():
        results[key][f'{metric_name}_mean'] = np.mean(metric_vals)
        results[key][f'{metric_name}_std'] = np.std(metric_vals)

# def simplify_columns(df):
#     column_stems = [c[:-len("_mean")] for c in df.columns if "_mean" in c]
#     for column in column_stems:
#         df[column] = df.apply(lambda x: f"{x[f'{column}_mean']:.1f} ± {x[f'{column}_std']:.1f}" if x[f'{column}_mean'] > 1 else
#                 f"{x[f'{column}_mean']:.2g} ± {x[f'{column}_std']:.2g}", axis=1)
#         df.drop(labels=[f"{column}_mean", f"{column}_std"], axis=1, inplace=True)
#     return df

def simplify_columns(df):
    column_stems = [c[:-len("_mean")] for c in df.columns if "_mean" in c]
    for column in column_stems:
        if is_numeric_dtype(df[f'{column}_mean']):
            df[column] = df.apply(lambda x: rf"${flf(x[f'{column}_mean'])} \pm {flf(x[f'{column}_std'])}$", axis=1)
        else:
            df[column] = df[f'{column}_mean']
        # df[column] = df.apply(lambda x: rf"${x[f'{column}_mean']:.2f} \pm {x[f'{column}_std']:.2f}$" if x[f'{column}_mean'] > 1 else
        #         rf"${x[f'{column}_mean']:.2g} \pm {x[f'{column}_std']:.2g}$", axis=1)
        df.drop(labels=[f"{column}_mean", f"{column}_std"], axis=1, inplace=True)
    return df

def flf(x):
    if x == 0:
        return "0"
    if isinstance(x, str):
        return x
    if .01 < x < 10000:
        x = f"{x:.1f}"
    else:
        x = f"{x:.1e}"
    if 'e' in x:
        b, e = x.split('e')
        return f'{b} \\times 10^{{{int(e)}}}'
    else:
        return x



main()
