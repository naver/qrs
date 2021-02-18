# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import streamlit as st
from pathlib import Path
import pickle
import pandas as pd
from numpy import floor, log10
from pandas.api.types import is_numeric_dtype
import numpy as np

def main():
    path = Path("<output-dir>/")

    ebms = {
        ("DPG", r"50\% female + 100\% science"): path/ "1M/female_science/direct-female_science.estimates.pkl",
        ("DPG", r"50\% female + 100\% sports"): path/ "1M/female_sports/direct-female_sports.estimates.pkl",
        ("DPG", r"50\% female"): path/ "1M/female/direct-female.estimates.pkl",
        ("DPG", r"100\% amazing"): path/ "1M/amazing/direct-amazing.estimates.pkl",
        ("DPG", r"100\% wikileaks"): path/ "1M/wikileaks/direct-wikileaks.estimates.pkl"
    }

    dfs = []
    for (prop, ebm),v in ebms.items():
        df = pickle.load(open(v, 'rb'))
        df = df[df['sampler'] == 'qrs']
        df['Proposal'] = prop
        df['EBM'] = ebm
        df['Z_mean'] = df['Z'].mean()
        df['Z_std'] = df['Z'].std()
        df = simplify_columns(df)
        for ar in [0.1, 3e-2, 1e-3]:
            df_flt = filter_by_closest_ar(df, ar)
            dfs.append(df_flt)
    df = pd.concat(dfs)
    mean_df = df.groupby(['EBM', 'Proposal', 'beta']).agg(mean_str).add_suffix('_mean')
    std_df = df.groupby(['EBM', 'Proposal', 'beta']).agg(std_str).add_suffix('_std')
    mean_std_df = pd.concat([mean_df, std_df], axis=1)
    mean_std_df = simplify_columns(mean_std_df)
    mean_std_df = mean_std_df[["ar", "tvd", "kl_p", "Z", "Z_beta"]]
    mean_std_df = mean_std_df.reset_index()
    mean_std_df = mean_std_df.sort_values(['EBM', 'Proposal', 'beta'])
    mean_std_df.beta = mean_std_df.beta.apply(lambda x: round(x, 3 - int(floor(log10(abs(x)))))).apply(flf).apply(lambda x:f'${x}$')
    mean_std_df1  = mean_std_df[['EBM', 'Proposal', 'beta', 'ar', 'tvd', 'kl_p']]
    mean_std_df2  = mean_std_df[['EBM', 'Proposal', 'beta', 'Z', 'Z_beta']]
    for df in [mean_std_df1, mean_std_df2]:
        df.rename(columns={"kl_p": r"$\KL(p, p_\beta)$",
            "tvd": r"$\TVD(p, p_\beta)$",
            "Z_beta": r"$Z_\beta$",
            "Z": "$Z$",
            "beta": r"$\beta$", 
            "ar": "AR",
            "EBM": "$P$",
            "Proposal": "$q$"}, inplace=True)
    st.write(mean_std_df1)
    st.write(mean_std_df2)
    st.code(mean_std_df1.to_latex(escape=False, index=False))
    st.code(mean_std_df2.to_latex(escape=False, index=False))

def filter_by_closest_ar(df, ar):
    i = df['ar'].sub(ar).abs().idxmin()
    beta = df.iloc[i]['beta']
    return df[df['beta'] == beta]


def simplify_columns(df):
    column_stems = [c[:-len("_mean")] for c in df.columns if "_mean" in c]
    for column in column_stems:
        if is_numeric_dtype(df[f'{column}_mean']):
            df[column] = df.apply(lambda x: rf"${flf(x[f'{column}_mean'])} \pm {flf(x[f'{column}_std'])}$", axis=1)
        else:
            df[column] = df[f'{column}_mean']
        df.drop(labels=[f"{column}_mean", f"{column}_std"], axis=1, inplace=True)
    return df

def flf(x):
    if isinstance(x, str):
        return x
    if .01 < x < 10000:
        x = f"{x:.1g}"
    else:
        x = f"{x:.1e}"
    if 'e' in x:
        b, e = x.split('e')

        return f'{b} \\times 10^{{{int(e)}}}'
    else:
        return x

def mean_str(col):
    if is_numeric_dtype(col):
        return col.mean()
    else:
        return col.unique() if col.nunique() == 1 else np.NaN

def std_str(col):
    if is_numeric_dtype(col):
        return col.std()
    else:
        return col.unique() if col.nunique() == 1 else np.NaN

main()
