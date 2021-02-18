# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import os, sys
base_dir = os.path.dirname(os.path.abspath(__file__))
lam1 = 10
lam2 = 9
results_dir = os.path.join(base_dir, f'results_{lam1}_{lam2}')
if not os.path.exists(results_dir):
        os.makedirs(results_dir)
from discontrol.distribution.poisson import PoissonDistribution
from discontrol.sampler import QuasiRejectionSampler, AccumulatorSampler, sample_until
from stats.goodness_of_fit import kl_exp_obs, tvd_exp_obs
from collections import Counter
from tqdm import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
custom_cycler = cycler(color=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
    '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'])
plt.rc('axes', prop_cycle=custom_cycler)

use_altair = False

def main():
    p = PoissonDistribution(lam=lam1)
    q = PoissonDistribution(lam=lam2, batch_size=100)
    res = []
    for N in (10**i for i in trange(2, 7)):
        samplers = {'optimal': p,
                'q': q,
                'QRS': QuasiRejectionSampler(p, q, batch_size=N),
                'QRS (mAR=1/2)': QuasiRejectionSampler(p, q, batch_size=N, min_acceptance_rate=0.5),
                'QRS (mAR=1/4)': QuasiRejectionSampler(p, q, batch_size=N, min_acceptance_rate=0.25),
        }
        for method_name, sampler in samplers.items():
            repetitions = 5
            for _ in range(repetitions):
                xs = sample_until(sampler, N)
                obs = Counter(xs)
                kl = kl_exp_obs(p, obs)
                tvd = tvd_exp_obs(p, obs)
                try:
                    ar = sampler.get_acceptance_rate()
                    beta = sampler.beta
                    tvd_bound = sampler.get_tvd_bound()
                except:
                    ar = 1
                    beta = None
                    tvd_bound = None
                res.append({'strategy': method_name, 'kl': kl, 'tvd': tvd, 'samples': N, 'ar': ar, 'beta': beta, 'tvd_bound': tvd_bound})
    df = pd.DataFrame(res)
    print(df)
    plot_column(df, 'kl', logy=True, logx=True,xlabel="$n$", ylabel="$KL(O||p)$")
    plot_column(df, 'tvd', logy=True, logx=True,xlabel="$n$", ylabel="$TVD(O||p)$")
    plot_column(df, 'ar', logy=True, logx=True,xlabel="$n$", ylabel='AR')
    df.to_csv(f'{results_dir}/qrs_fit.csv')


def plot_column(df, column_name, logy=False, logx=False, xlabel=None, ylabel=None):
    out_fn = os.path.join(results_dir, column_name + '.png')
    if use_altair:
        import altair as alt
        alt.Chart(df).mark_line().encode(
                x=alt.X('samples', scale=alt.Scale(type='log' if logx else 'linear')),
                y=alt.Y(column_name, scale=alt.Scale(type='log' if logy else 'linear')),
                color='strategy').save(out_fn)
    else:
        df = df[~df[column_name].isna()]
        plt.clf()
        fig = sns.lineplot(data=df, x="samples", y=column_name, hue='strategy', style='strategy')
        if column_name == 'tvd':
            aux=df[df['strategy'] == 'QRS']
            plt.plot(aux['samples'], aux['tvd_bound'], color='grey', linestyle='--', label='QRS bound ($\delta=0.05$)')
        plt.legend()
        fig.set(xscale='log')
        fig.set(yscale='log')
        if ylabel:
            plt.ylabel(ylabel)
        if xlabel:
            plt.xlabel(xlabel)
        plt.tight_layout()
        plt.savefig(out_fn)

if __name__ == '__main__':
    main()
