# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import streamlit as st
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


sns.set_palette(sns.color_palette("colorblind"))
style = {'s': 100, 'alpha': 0.2}
markers_style = {'s': 100}
line_style = {"ci": 'sd', "err_style": "band", "legend": False}
MARKER_SIZE=100
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
         '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

def main():
    paths = {
            "1M": "<output-dir>/1M",
            "10M": "<output-dir>/10M",
            "prompts": "<output-dir>/prompts/wikileaks"
    }
    sampleset_name = st.selectbox("Total number of samples", paths.keys())
    path = Path(paths[sampleset_name])

    if sampleset_name in ["1M", "10M"]:
        experiments = [d.name for d in path.glob("*")]
        experiment = st.selectbox("Experiment", experiments)
        df = pickle.load(open(path / experiment / f"direct-{experiment}.estimates2.pkl", "rb"))
    else:
        experiments = [d.stem for d in path.glob("*.jsonl")]
        experiment = st.selectbox("Experiment", experiments)
        df = pickle.load(open(path / f"{experiment}.estimates.pkl", "rb"))

    figures_dir = Path(f"figures/") / sampleset_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = df.reset_index()
    df.loc[df['sampler'] == 'qrs','order'] = 0
    df.loc[df['sampler'] == 'proposal','order'] = 1
    df.loc[df['sampler'] == 'naive','order'] = 2
    st.write(df)
    plot_format = st.radio('Plot format', ['single', 'multi'])

    if experiment not in ['amazing', 'wikileaks']:
        df = df[df['sampler'] != 'naive']

    df = df.sort_values(['sampler', 'ar'])
    unit_line = np.logspace(np.log(df['ar'].min()), 1e-10, base=np.e, endpoint=True)
    if 'constraint-Wikileaks' in df:
        df['constraint-Wikileaks'] = np.minimum(1, df['constraint-Wikileaks'])
    df['binned_ar'] = df.groupby(['sampler','beta'])['ar'].transform('mean')
    mean_df = df.groupby(['sampler', 'binned_ar']).mean().add_suffix('_mean')
    std_df = df.groupby(['sampler', 'binned_ar']).std().add_suffix('_std')
    mean_std_df = pd.concat([mean_df, std_df], axis=1)
    mean_std_df = simplify_columns(mean_std_df)
    st.write("## Aggregate statistics")
    st.write(mean_std_df)
    st.code(mean_std_df.to_latex(longtable=True, escape=False))
    mean_df = mean_df.reset_index()
    mean_df = mean_df.sort_values('order_mean')
    st.write(mean_df)
    df = df.sort_values('order')

    ### Divergence Metrics
    st.write("## Divergence Metrics")

    plots = [
            #(plot_ar, {'savefig': figures_dir / f"{experiment}-AR.png"}),
            (plot_tvd, {'savefig': figures_dir / f"{experiment}-TVD.png"}),
            (plot_kl_p, {'savefig': figures_dir / f"{experiment}-KL-p.png"}),
            (plot_kl_a, {'savefig': figures_dir / f"{experiment}-KL-a.png"})]
    targets = {'constraint-amazing': 100.,
            'constraint-Wikileaks': 100.,
            'constraint-science': 100.,
            'constraint-sports': 100.,
            'constraint-female': 50}
    for c in df:
        if c.startswith('constraint-'):
            plots.append((plot_constraint_satisfaction, {'name': c, 'savefig': figures_dir / f"{experiment}-{c}.png", 'target': targets[c] }))


    if plot_format == 'single':
        for plot_f, args in plots:
            args['savefig'] = None
            fig = plot_f(df, mean_df.copy(), experiment=experiment, **args)
            st.pyplot(fig)
    else:
        fig, axs = plt.subplots(1, len(plots), figsize=(3*len(plots), 4))
        for ax, (plot_f, args) in zip(axs, plots):
            args['savefig'] = None
            plot_f(df, mean_df.copy(), experiment=experiment, ax=ax, **args)
        plt.tight_layout()
        st.pyplot(fig)
        savefig = figures_dir / f"QRS-{experiment}-with-variance.png"
        if savefig:
            st.write(f"Saved to {savefig}")
            fig.savefig(savefig, dpi=300, bbox_inches='tight')
    


def plot_kl_a(df, mean_df, experiment=None, savefig=None, ax=None, legend_pos=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    mean_df['sampler'] = mean_df['sampler'].str.replace('qrs', r'$D_\mathrm{KL}(p_\\beta, a)$')
    mean_df['sampler'] = mean_df['sampler'].str.replace('proposal', r'$D_\mathrm{KL}(q, a)$')
    mean_df['sampler'] = mean_df['sampler'].str.replace('naive', r'$D_\mathrm{KL}(q_{proj}, a)$')
    sns.scatterplot(data=mean_df, x="binned_ar", y="kl_a_mean", ax=ax, style="sampler", hue="sampler", **markers_style)
    sns.lineplot(data=df, x="binned_ar", y="kl_a", ax=ax, style="sampler", hue="sampler", **line_style)
    ax.set_xlabel("acceptance rate")
    ax.set(xscale='log')
    ax.set_ylabel(r"$D_\mathrm{KL}(\cdot, a)$", fontsize=13)
    ax.invert_xaxis()
    ax.yaxis.set_major_formatter('{x:0<4.1f}')
    clean_legend(ax)
    if fig is not None:
        if savefig:
            fig.savefig(savefig, dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
        return fig

    
def plot_kl_p(df, mean_df, experiment=None, savefig=None, ax=None, legend_pos=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    mean_df['sampler'] = mean_df['sampler'].str.replace('qrs', r'$D_\mathrm{KL}(p, p_\\beta)$')
    mean_df['sampler'] = mean_df['sampler'].str.replace('proposal', r'$D_\mathrm{KL}(p, q)$')
    mean_df['sampler'] = mean_df['sampler'].str.replace('naive', r'$D_\mathrm{KL}(p, q_{proj})$')
    sns.scatterplot(data=mean_df, x="binned_ar", y="kl_p_mean", ax=ax, style="sampler", hue="sampler", **markers_style)
    sns.lineplot(data=df, x="binned_ar", y="kl_p", ax=ax, style="sampler", hue="sampler", **line_style)
    ax.set_xlabel("acceptance rate")
    ax.set(xscale='log')
    ax.set_ylabel(r"$D_\mathrm{KL}(p,\cdot)$", fontsize=13)
    ax.invert_xaxis()
    clean_legend(ax)
    if fig is not None:
        if savefig:
            fig.savefig(savefig, dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
        return fig

def plot_tvd(df, mean_df, experiment=None, savefig=None, ax=None, legend_pos=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    mean_df['sampler'] = mean_df['sampler'].str.replace('qrs', r'$TVD(p, p_\\beta)$')
    mean_df['sampler'] = mean_df['sampler'].str.replace('proposal', r'$TVD(p, q)$')
    mean_df['sampler'] = mean_df['sampler'].str.replace('naive', r'$TVD(p, q_{proj})$')
    sns.scatterplot(data=mean_df, x="binned_ar", y="tvd_mean", ax=ax, style="sampler", hue="sampler", **markers_style)
    sns.lineplot(data=df, x="binned_ar", y="tvd_ub", ax=ax, label=r"$1 - p(A_\beta)$", color=colors[3])
    sns.lineplot(data=df, x="binned_ar", y="tvd", ax=ax, style="sampler", hue="sampler", **line_style)

    ax.set_ylabel('TVD', fontsize=13)
    ax.set(xscale='log')
    ax.set_xlabel("acceptance rate")
    ax.invert_xaxis()
    clean_legend(ax)
    if fig is not None:
        if savefig:
            fig.savefig(savefig, dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
        return fig

def plot_ar(df, mean_df, experiment=None, savefig=None, ax=None, legend_pos=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    print([(k,len(v)) for k,v in df.items()])
    sns.lineplot(data=df, x="beta", y="ar", ax=ax, style="sampler", hue="sampler", markers=False, dashes=False, legend=False)
    sns.scatterplot(data=df, x="beta", y="ar", ax=ax, style="sampler", hue="sampler", **style)
    ax.set_ylabel('AR', fontsize=13)
    ax.set(yscale='log')
    ax.set_xlabel(r"$\beta$")
    clean_legend(ax)
    if fig is not None:
        if savefig:
            fig.savefig(savefig, dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
        return fig

def plot_constraint_satisfaction(df, mean_df, name, target, observed_values=None, experiment=None, savefig=None, ax=None, legend_pos=None, show_ebm=True):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    df[name] = 100 * df[name]
    mean_df[f"{name}_mean"] = 100 * mean_df[f"{name}_mean"]
    ebm = df[df['sampler'] == 'proposal']['p-'+name].mean() * 100
    mean_df['sampler'] = mean_df['sampler'].str.replace('qrs', r'$p_\\beta$')
    mean_df['sampler'] = mean_df['sampler'].str.replace('proposal', r'$q$')
    mean_df['sampler'] = mean_df['sampler'].str.replace('naive', r'$q_{proj}$')
    graph = sns.scatterplot(data=mean_df, x="binned_ar", y=f"{name}_mean", ax=ax, style="sampler", hue="sampler", **markers_style)
    sns.lineplot(data=df, x="binned_ar", y=name, ax=ax, style="sampler", hue="sampler", **line_style)
    graph.axhline(target, linestyle="dashed", label="target moments", color=colors[-3])
    graph.axhline(ebm, linestyle="dashed", label="p", color=colors[-4])
    ax.set_xlabel("acceptance rate")
    ax.set_ylabel(f"{name} %", fontsize=13)
    ax.set(xscale='log')
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]+0.03)
    ax.invert_xaxis()
    clean_legend(ax)
    if savefig:
        fig.savefig(savefig, dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
    return fig

def clean_legend(ax):
    ax.legend()
    return

def simplify_columns(df):
    column_stems = [c[:-len("_mean")] for c in df.columns if "_mean" in c]
    for column in column_stems:
        df[column] = df.apply(lambda x: rf"${x[f'{column}_mean']:.2f} \pm {x[f'{column}_std']:.2f}$" if x[f'{column}_mean'] > 1 else
                rf"${x[f'{column}_mean']:.2g} \pm {x[f'{column}_std']:.2g}$", axis=1)
        df.drop(labels=[f"{column}_mean", f"{column}_std"], axis=1, inplace=True)
    return df

main()
