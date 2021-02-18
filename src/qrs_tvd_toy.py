# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import os, sys
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, f'results_qrs_tvd')
if not os.path.exists(results_dir):
        os.makedirs(results_dir)
from discontrol.distribution.poisson import PoissonDistribution
from discontrol.sampler import QuasiRejectionSampler, AccumulatorSampler, sample_until
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import streamlit as st
sns.set_palette(sns.color_palette("colorblind"))
C_bound = 'C6'
C_ar = 'C7'
C_tvd = 'C0'
C_q = 'C1'

ci='sd'

def main():
    st.title('QRS TVD')
    samples_exp = st.slider(label='Number of q samples (in log_10 base)', 
            min_value=2, max_value=7, value=5)
    lam1 = st.number_input(label='λ1', min_value=1, max_value=20, value=11)
    lam2 = st.number_input(label='λ2', min_value=1, max_value=20, value=10)
    num_experiments = st.slider(label='Number of independent experiments',
            value=10, min_value=1, max_value=20)
    min_beta = st.number_input(label='Beta min', value=0.5)
    max_beta = st.number_input(label='Beta max', value=3.5)
    delta = st.number_input(label='δ', min_value=0., max_value=1., value=0.05)

    N = 10 ** samples_exp
    p = PoissonDistribution(lam=lam1)
    q = PoissonDistribution(lam=lam2, batch_size=N)
    betas = np.arange(min_beta, max_beta, 0.05)
    df_data, all_betas = get_tvd_estimates(p, q, betas, num_experiments)
    df_data = add_bounds(df_data, betas, N, delta)
    plot_format = st.radio('Plot format', ['single', 'multi'])
    plots = [plot_kl, plot_ar, plot_kl_ar]
    fig = make_plots(df_data, plots, single=(plot_format=='single'))
    if plot_format == 'multi':
        fig.savefig(f'figures/QRS-poisson-{lam1}-{lam2}-10e{samples_exp}-{len(plots)}-KL.png', dpi=300, bbox_inches='tight')
    st.write('## Distribution of $\\beta$ values')
    plt.clf()
    fig, ax = plt.subplots()
    plt.xlabel('Beta')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.hist(all_betas, label='Beta distribution')
    st.pyplot(fig)

def get_tvd_estimates(p, q, beta_ticks, repetitions):
    data = []
    all_betas = []
    for i in range(repetitions):
        x = q.sample()
        log_p_x = np.array(p.log_score(x))
        log_q_x = np.array(q.log_score(x))
        p_x = np.exp(log_p_x)
        q_x = np.exp(log_q_x)
        beta_x = np.exp(log_p_x - log_q_x)
        q_tvd = np.mean(np.abs(beta_x -1)) / 2
        q_kl = np.mean(beta_x * np.log(beta_x))
        all_betas.append(beta_x)
        print(beta_x.max())
        for beta in beta_ticks:
            betas_A_or_0 = beta_x * (beta_x < beta)
            tvd_bound = 1 - betas_A_or_0.mean()
            log_beta = np.log(beta)
            log_P_beta_x = np.minimum(log_p_x, log_beta + log_q_x)
            P_beta_x = np.minimum(p_x, beta * q_x)
            if (P_beta_x == p_x).all():
                Z_beta = 1 # avoid numerical errors
            else:
                Z_beta = np.mean(np.exp(log_P_beta_x - log_q_x))
            ar = Z_beta / beta
            log_p_beta_x = log_P_beta_x - np.log(Z_beta)
            p_beta_x = P_beta_x / Z_beta
            tvd = np.abs(np.array(beta_x - p_beta_x / q_x)).mean() / 2
            kl = max(0, np.mean(beta_x * (log_p_x - log_P_beta_x)) + np.log(Z_beta))
            data.append({'beta': beta, 'tvd_bound': tvd_bound, 'tvd': tvd, 'ar': ar, 'q_tvd': q_tvd, 'kl': kl, 'q_kl': q_kl})
        df_data = pd.DataFrame(data)
    print(df_data)
    return df_data, np.concatenate(all_betas)

def make_plots(df_data, plot_funcs, single=False):
    if single:
        figs = []
        for plot_func in plot_funcs:
            fig, ax = plt.subplots()
            plot_func(df_data, ax)
            figs.append(fig)
            st.pyplot(fig)
        return figs
    else:
        fig, axs = plt.subplots(1, len(plot_funcs), figsize=(3 * len(plot_funcs), 4.5))
        for ax, plot_func in zip(axs, plot_funcs):
            plot_func(df_data, ax)
        plt.tight_layout()
        st.pyplot(fig)
        return fig


def plot_tvd(df_data, ax):
    sns.lineplot(data=df_data, x="beta", y='tvd_bound', label=r"$1 - p(A_\beta)$", color=C_bound, ax=ax, ci=ci)
    sns.lineplot(data=df_data, x="beta", y='tvd', label=r"TVD($p$, $p_\beta$)", color=C_tvd, marker='o', ax=ax, ci=ci)
    print(df_data['q_tvd'])
    sns.lineplot(data=df_data, x="beta", y="q_tvd", label="TVD($p$, $q$)", linewidth='2', linestyle=':', color=C_q, ax=ax, ci=ci)
    ax.set_ylabel('TVD', fontsize=14)
    ax.set_xlabel("$\\beta$", fontsize=14)
    ax.legend(fontsize=14, loc='upper right')
    plt.tight_layout()

def plot_kl(df_data, ax):
    sns.lineplot(data=df_data, x="beta", y='kl', label=r"$D_\mathrm{KL}(p, p_\beta)$", color=C_tvd, marker='o', ax=ax, ci=ci)
    sns.lineplot(data=df_data, x="beta", y="q_kl", label=r"$D_\mathrm{KL}(p, q)$", linewidth='2', linestyle=':', color=C_q, ax=ax, ci=ci)
    ax.set_ylabel(r'$D_\mathrm{KL}(p, \cdot$)', fontsize=14)
    ax.set_xlabel("$\\beta$", fontsize=14)
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()


def plot_ar(df_data, ax):
    sns.lineplot(data=df_data, x="beta", y='ar', color=C_ar, ax=ax)
    ax.set_ylabel('acceptance rate', fontsize=14)
    #ax.set(yscale='log')
    ax.set_xlabel("$\\beta$", fontsize=14)
    plt.tight_layout()

def plot_tvd_ar(df_data, ax):
    x = np.linspace(0, 1)

    df_data['binned_ar'] = df_data.groupby(['beta'])['ar'].transform('mean')
    sns.lineplot(data=df_data, x="binned_ar", y='tvd_bound', legend=False, label=r"$1 - p_\beta(A)$", ax=ax, color=C_bound, ci=ci)
    sns.lineplot(data=df_data, x="binned_ar", y='tvd', ax=ax, label=r"TVD($p$, $p_\beta$)", marker="o", color=C_tvd, ci=ci)
    ax.scatter(1, df_data['q_tvd'].min(), label='TVD($p$, $q$)', marker='X', color=C_q, s=100)
    ax.set_ylabel('TVD', fontsize=14)
    ax.set_xlabel("acceptance rate", fontsize=14)
    ax.invert_xaxis()
    ax.legend(fontsize=14, loc='upper right')
    plt.tight_layout()

def plot_kl_ar(df_data, ax):
    x = np.linspace(0, 1)
    df_data['binned_ar'] = df_data.groupby(['beta'])['ar'].transform('mean')
    sns.lineplot(data=df_data, x="binned_ar", y='kl', ax=ax, label=r"$D_\mathrm{KL}(p,p_\beta)$", marker='o', color=C_tvd, ci=ci)
    ax.scatter(1, df_data['q_kl'].mean(), label=r'$D_\mathrm{KL}(p, q)$', marker='X', color=C_q, s=100)
    ax.set_ylabel(r'$D_\mathrm{KL}(p, \cdot)$', fontsize=14)
    ax.set_xlabel("acceptance rate", fontsize=14)
    ax.invert_xaxis()
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()


def add_bounds(df_data, betas, N, delta=0.05):
    for beta in betas:
        bound = np.sqrt(beta ** 2 / N / 2 * (-np.log(delta))) 
        df_data.loc[df_data['beta'] == beta,'tvd_hoeffding_top'] = \
                df_data.loc[df_data['beta'] == beta, 'tvd_bound'] + bound
        df_data.loc[:, 'tvd_hoeffding_top'].clip(upper=1, inplace=True)
        df_data.loc[df_data['beta'] == beta,'tvd_hoeffding_bottom'] = \
                df_data.loc[df_data['beta'] == beta, 'tvd_bound'] - bound
        df_data.loc[:, 'tvd_hoeffding_bottom'].clip(lower=0, inplace=True)
    return df_data


if __name__ ==  '__main__':
    main()
