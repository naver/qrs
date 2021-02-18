# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import os, sys
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, f'results_qrs_tvd')
if not os.path.exists(results_dir):
        os.makedirs(results_dir)
from discontrol.distribution.lm import GDCAutoregressiveLM, HFAutoregressiveLM
from discontrol.sampler import QuasiRejectionSampler, AccumulatorSampler, sample_until
from discontrol.scorer import SingleWordFeature
from discontrol import misc
sys.modules['misc'] = misc
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import streamlit as st
from itertools import *
import operator
from discontrol.scorer.common import broadcast
import torch

custom_cycler = cycler(color=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
    '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'])
plt.rc('axes', prop_cycle=custom_cycler)
def main():
    st.title('QRS TVD')
    experiment_name = st.selectbox(label='Experiment', options=['amazing'])
    x, log_P_x, log_q_x = load_samples(experiment_name)
    num_samples = st.slider(max_value=10000000, value=100000, label='Number of samples')
    x, log_P_x, log_q_x = x[-num_samples:], log_P_x[-num_samples:], log_q_x[-num_samples:]
    scorers = get_scorers(experiment_name)
    s_x = {feature_name: np.array(scorer.score(x)) for feature_name, scorer in scorers.items()}
    bin_funcs = {}
    log_unnorm_beta_x = np.exp(log_P_x - log_q_x)
    Z = np.log(log_unnorm_beta_x.mean())
    log_p_x = log_P_x - Z
    beta_x = np.exp(log_p_x - log_q_x)
    min_beta = st.number_input(label='Beta min', value=0.05)
    max_beta = st.number_input(label='Beta max', value=2000)
    beta_tick_size = st.number_input(label='Beta tick size', value=float(beta_x.max() - beta_x.min())/100)
    tot_covered = ((min_beta <= beta_x) & (beta_x <= max_beta)).sum().item()
    st.write(f'{tot_covered} of {len(beta_x)} ({tot_covered/len(beta_x)*100:.2f}%) samples in A')
    beta_ticks = np.arange(min_beta, max_beta, beta_tick_size)
    df_data = get_tvd_estimates(beta_x, beta_ticks)
    delta = st.number_input(label='δ', min_value=0., max_value=1., value=0.05)
    df_data = add_bounds(df_data, beta_ticks, len(log_P_x), delta)
    st.write('## Metrics as a function of $\\beta$')
    plot_metrics_as_function_of_beta(df_data, scorers.keys())
    st.write('## TVD/AR tradeoff with varying $\\beta$')
    point_estimates = {}
    naive_tvd, naive_ar = get_naive_tvd_ar(log_p_x, log_q_x)
    point_estimates['naive (directly computed)'] = (naive_ar, naive_tvd)
    naive_samples = [x_i for x_i, s_i in zip(x, next(iter(s_x.values()))) if s_i]
    print('naive AR (sanity check)', len(naive_samples)/len(x))
    for bin_method_name, (nbins, bin_func) in bin_funcs.items():
        point_estimates[bin_method_name] = (naive_ar, get_binned_tvd(log_p_x, log_q_x, x, naive_samples, bin_func, nbins))
    fig = plot2(df_data, point_estimates)
    st.pyplot(fig)
    st.write('## Distribution of $\\beta$ values')
    plt.clf()
    fig, ax = plt.subplots()
    plt.xlabel('Beta')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.hist(beta_x, label='Beta distribution')
    st.pyplot(fig)
    st.write('## Naive Filter Samples')
    st.write(naive_samples)
    st.write(list(map(operator.itemgetter(1), islice(filter(lambda t: t[0], zip(next(iter(s_x.values())), x)), 100))))


@st.cache
def load_samples(experiment_name):
    x = pickle.load(open(f'<output-dir>/{experiment_name}/old/direct-{experiment_name}.pkl', 'rb'))
    log_P_x = pickle.load(open(f'<output-dir>/{experiment_name}/old/direct-{experiment_name}.ebm.pkl', 'rb'))
    log_q_x = pickle.load(open(f'<output-dir>/{experiment_name}/old/direct-{experiment_name}.proposal.pkl', 'rb'))
    return x, log_P_x, log_q_x

def get_scorers(experiment_name):
    scorers = {
            'amazing': 
            {'amazing occurs': SingleWordFeature("amazing")}
    }
    return scorers[experiment_name]

def get_bins(experiment_name):
    if experiment_name == 'amazing':
        nbins = st.slider(label='Num. of hand-picked bins', value=3, min_value=3, max_value=10)
        nsoftbins, softbins = torch.load(open('bins.pkl', 'rb'))
        return {
                'naive (binned by word count)': (nbins, broadcast(lambda x: min(x.count('amazing'), nbins))),
                'naive (learned bins)': (nsoftbins, lambda xs: softbins(xs).max(dim=1).indices.cpu().numpy()) # TODO: batch samples!!
        }
    else:
        return {}

def get_tvd_estimates(beta_x, beta_ticks):
    data = []
    Z = beta_x.mean()
    for beta in beta_ticks:
        betas_A_or_0 = beta_x * (beta_x < beta)
        tvd = 1 - betas_A_or_0.mean() / Z
        r_x = np.minimum(1, beta_x / beta)
        ar = r_x.mean()
        row = {'beta': beta, 'tvd': tvd, 'ar': ar}
        data.append(row)
    df_data = pd.DataFrame(data)
    print(df_data)
    return df_data

def get_naive_tvd_ar(log_p_x, log_q_x):
    rejected = np.isinf(log_p_x)
    accepted = ~rejected
    ar = accepted.sum() / len(log_p_x)
    log_sigma_x = log_q_x - np.log(ar)
    log_sigma_x[rejected] = np.NINF
    tvd = 0.5 * np.mean(np.abs(np.exp(log_sigma_x - log_q_x) - np.exp(log_p_x - log_q_x)))
    return tvd, ar


def get_binned_tvd(log_p_x, log_q_x, x, x_prime, bin_func, nbins):
    bins_p = np.zeros(nbins)
    bins_count = np.zeros(nbins)
    bin_x = np.array(bin_func(x))
    bin_x_prime = np.array(bin_func(x_prime))
    betas_x = np.exp(log_p_x - log_q_x)
    for b in range(0, nbins):
        bins_p[b] = (betas_x * (bin_x == b)).mean()
        bins_count[b] = (bin_x_prime == b).sum()
    bins_p /= bins_p.sum()
    bins_count /= bins_count.sum()
    st.write('empirical', bins_count)
    st.write('projected', bins_p)
    return np.max(np.abs(bins_count - bins_p))


def plot_metrics_as_function_of_beta(df_data, feature_names):
    plot_tvd_as_function_of_beta(df_data)
    plot_metric_as_function_of_beta(df_data, 'ar', 'Acceptance Rate')

def plot_metric_as_function_of_beta(df_data, metric, label):
    fig, ax = plt.subplots()
    sns.lineplot(data=df_data, x="beta", y=metric, color='#ff7f00')
    ax.set_ylabel(label)
    ax.set(yscale='log', xscale='log')
    ax.set_xlabel("$\\beta$", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

def plot_tvd_as_function_of_beta(df_data):
    fig, ax = plt.subplots()
    sns.lineplot(data=df_data, x="beta", y='tvd', ax=ax, color='#377eb8')
    sns.lineplot(data=df_data, x="beta", y='tvd_with_bound', ax=ax, color='#377eb8')
    ax.lines[1].set_linestyle(':')
    ax.set_ylabel('TVD bound')
    ax.set(yscale='log', xscale='log')
    ax.set_xlabel("$\\beta$", fontsize=14)
    plt.legend(handles=[ax.lines[0], ax.lines[1]],
            labels=[r"1 - $\hat{p}_\beta(A)$", r"1 - $\hat{p}_\beta(A) + \epsilon$"])
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

def plot2(df_data, point_estimates):
    fig, ax = plt.subplots()
    sns.lineplot(data=df_data, x="ar", y='tvd', ax=ax,  label='QRS bound')
    sns.lineplot(data=df_data, x="ar", y='tvd_with_bound', ax=ax,  label='QRS Hoeffding bound')
    for bin_method_name, (ar, tvd) in point_estimates.items():
        plt.plot([ar], [tvd], marker='x', linestyle='None', markersize=10, label=bin_method_name)
        # follow https://stackoverflow.com/questions/65541863/pyplot-adding-point-projections-on-axis
        plt.annotate(f'{tvd:.1e}', xy=(ar, tvd), xytext=(0, tvd), 
                     textcoords=plt.gca().get_yaxis_transform(),
                     arrowprops={'arrowstyle': '-', 'ls':'--', 'color': 'grey'}, 
                     size=8,
                     va='center', ha='right') 
        projected_ar, projected_tvd = project_into_curve_x(df_data, tvd, 'tvd', 'ar')
        print(projected_ar, projected_tvd)
        plt.plot([ar, projected_ar], [projected_tvd, projected_tvd], lw=1, ls='--', color='grey')
        plt.annotate(f'{projected_ar:.1e}', xy=(projected_ar, projected_tvd),
                     xytext=(projected_ar, -0.05), 
                     textcoords=plt.gca().get_xaxis_transform(),
                     arrowprops={'arrowstyle': '-', 'ls':'--', 'color': 'grey'},
                     size=8,
                     va='top', ha='center') 
    ax.set_ylabel('TVD')
    ax.set(yscale='log')
    ax.set(xscale='log')
    ax.set_xlabel("acceptance rate")
    ax.invert_xaxis()
    plt.legend()
    plt.tight_layout()
    return fig

def project_into_curve_x(df_data, y, y_metric, x_metric):
    closest_ind = find_closest(y, df_data, y_metric)
    return (df_data.loc[closest_ind, x_metric],
            df_data.loc[closest_ind, y_metric])


def find_closest(value, df, colname):
    # adapted from https://stackoverflow.com/questions/30112202/how-do-i-find-the-closest-values-in-a-pandas-series-to-an-input-number
    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return exactmatch.index
    else:
        try:
            lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
            return lowerneighbour_ind
        except ValueError:
            return df[colname].idxmin()

def add_bounds(df_data, betas, N, delta=0.05):
    for beta in betas:
        bound = np.sqrt(beta ** 2 / N / 2 * (-np.log(delta))) 
        df_data.loc[df_data['beta'] == beta,'tvd_with_bound'] = \
                np.minimum(1, df_data.loc[df_data['beta'] == beta, 'tvd'] + bound)
    return df_data


if __name__ ==  '__main__':
    main()
