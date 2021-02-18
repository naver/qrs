# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import argparse
import pickle
import random
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm, trange

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('samples_fn', type=Path)
    ap.add_argument('--beta-min', type=float, required=True)
    ap.add_argument('--beta-max', type=float, required=True)
    ap.add_argument('--n-betas', type=int, default=50)
    ap.add_argument('--n-samples', type=int, default=None, 
            help='number of samples used in estimating one single datapoint'
            ' (for subsampling and independent methods only)')
    ap.add_argument('--n-points-per-beta', type=int, default=20_000,
            help='number of independent estimations computed for every data point')
    ap.add_argument('--method', choices=['bootstrap', 'subsampling', 'independent'], default='subsampling')
    args = ap.parse_args()
    
    print('Loading EBM scores...')
    log_P_x = pickle.load(open(args.samples_fn.with_suffix('.ebm.pkl'), 'rb'))
    print('Loading proposal scores...')
    log_q_x = pickle.load(open(args.samples_fn.with_suffix('.proposal.pkl'), 'rb'))
    try:
        print('Loading base LM scores...')
        log_a_x = pickle.load(open(args.samples_fn.with_suffix('.base.pkl'), 'rb'))
    except:
        log_a_x = None
    print('Loading features...')
    phi_x = pickle.load(open(args.samples_fn.with_suffix('.features.pkl'), 'rb'))

    n_points_per_beta = args.n_points_per_beta
    print("QRS sampler (means)")
    betas = get_betas(args.n_betas, args.beta_min, args.beta_max)
    df_qrs = repeat_estimation_with_betas(betas, 'full', 1, args.n_samples, log_P_x, log_q_x, log_a_x, phi_x)
    df_qrs['sampler'] = 'qrs'
    df_qrs['estimator'] = 'full'
    print("QRS sampler (variance)")
    #betas = np.repeat(get_betas(args.n_betas, args.beta_min, args.beta_max), 
    #        n_points_per_beta)
    df_qrs_var = repeat_estimation_with_betas(betas, args.method, n_points_per_beta, args.n_samples, log_P_x, log_q_x, log_a_x, phi_x)
    df_qrs_var['sampler'] = 'qrs'
    df_qrs_var['estimator'] = args.method
    print("Naive sampler")
    df_naive = repeat_estimation_with_function(get_metrics_estimates_naive_filter,
            'full', n_points_per_beta, args.n_samples, log_P_x, log_q_x, log_a_x, phi_x)
    df_naive['sampler'] = 'naive'
    df_naive['estimator'] = 'full'
    print("Naive sampler (variance)")
    df_naive_var = repeat_estimation_with_function(get_metrics_estimates_naive_filter,
            args.method, n_points_per_beta, args.n_samples, log_P_x, log_q_x, log_a_x, phi_x)
    df_naive_var['sampler'] = 'naive'
    df_naive_var['estimator'] = args.method
    print("proposal")
    df_proposal = repeat_estimation_with_function(
            get_metrics_estimates_proposal,
            'full',
            n_points_per_beta, args.n_samples, log_P_x, log_q_x, log_a_x, phi_x)
    df_proposal['sampler'] = 'proposal'
    df_proposal['estimator'] = 'full'
    print("proposal (variance)")
    df_proposal_var = repeat_estimation_with_function(
            get_metrics_estimates_proposal,
            args.method,
            n_points_per_beta, args.n_samples, log_P_x, log_q_x, log_a_x, phi_x)
    df_proposal_var['sampler'] = 'proposal'
    df_proposal_var['estimator'] = args.method
    df = pd.concat([df_qrs, df_qrs_var, df_naive, df_naive_var, df_proposal, df_proposal_var])
    pickle.dump(df, open(args.samples_fn.with_suffix(f'.estimates.{args.method}.{args.n_betas}_betas.{args.n_points_per_beta}_points.{args.n_samples}_samples.pkl'), 'wb'))

def get_betas(n, min_val, max_val):
    return np.logspace(np.log(min_val), np.log(max_val), num=n, base=np.e)

def repeat_estimation_with_betas(betas, method, n_iterations, n_samples, log_P_x, log_q_x, log_a_x, phi_x):
    points = []
    for i in range(n_iterations):
        for beta in tqdm(betas):
            log_P_xi, log_q_xi, log_a_xi, phis_xi = bootstrap_resample(method, i, n_samples,
                    log_P_x, log_q_x, log_a_x, phi_x)
            point = get_metrics_estimates_at_beta(beta, log_P_xi, log_q_xi, log_a_xi, phis_xi)
            points.append(point)
    return pd.DataFrame(points)

def bootstrap_resample(method, i, n_samples, log_P_x, log_q_x, log_a_x, phi_x):
    N = len(log_P_x)
    # xi = np.random.choice(N, n_samples) # boostrap m-out-of-N
    if method == 'bootstrap':
        xi = np.random.choice(N, N) # standard bootstrap
    elif method == 'subsampling':
        xi = random.sample(range(N), n_samples) # subsampling
    elif method == 'independent':
        xi = np.arange(N)[n_samples*i:n_samples*(i+1)]
    elif method == 'full':
        xi = np.arange(N)
    log_P_xi = log_P_x[xi]
    log_q_xi = log_q_x[xi]
    if log_a_x is not None:
        log_a_xi = log_a_x[xi]
    else:
        log_a_xi = None
    phis_xi = phi_x.loc[xi]
    return log_P_xi, log_q_xi, log_a_xi, phis_xi

def get_metrics_estimates_at_beta(beta, log_P_xi, log_q_xi, log_a_xi, phis_xi):
    Z = np.mean(np.exp(log_P_xi - log_q_xi))
    log_Z = np.log(Z)
    log_p_xi = log_P_xi - np.log(Z) 
    r_xi = np.where(np.isneginf(log_P_xi), 0., np.minimum(1, np.exp(log_P_xi - safer_log(beta) - log_q_xi)))
    Z_K = np.mean(r_xi)
    log_Z_K = np.log(Z_K)
    log_K_xi = np.minimum(log_q_xi, log_P_xi - safer_log(beta))
    log_k_xi = log_K_xi - log_Z_K
    betas_xi = np.exp(log_P_xi - log_q_xi)
    betas_A_or_0 = betas_xi * (betas_xi < beta)
    tvd_ub = 1 - betas_A_or_0.mean() / Z
    tvd = 0.5 * np.mean(np.where(np.isneginf(log_P_xi), 0., np.abs(np.exp(log_k_xi - log_q_xi) - np.exp(log_p_xi - log_q_xi))))
    kl_p = np.mean(np.where(np.isneginf(log_P_xi), 0., np.exp(log_p_xi - log_q_xi) * (log_p_xi - log_k_xi)))
    if log_a_xi is not None:
        kl_a = np.mean(np.where(np.isneginf(log_P_xi), 0., np.exp(log_k_xi - log_q_xi) * (log_k_xi - log_a_xi)))
    else:
        kl_a = None
    estimates = {
            'beta': beta,
            'ar': Z_K,
            'tvd': tvd,
            'kl_p': kl_p,
            'kl_a': kl_a,
            'tvd_ub': tvd_ub,
            'Z': Z,
            'log_Z': log_Z,
            'Z_beta': Z_K,
            'log_Z_beta': log_Z_K,
            'Z:Z_beta': np.exp(log_Z - log_Z_K)
            }
    for name in phis_xi:
        phi_xi = phis_xi[name]
        cs = np.mean(np.exp(log_k_xi - log_q_xi) * phi_xi)
        estimates[f"constraint-{name}"] = cs
    return estimates

def repeat_estimation_with_function(get_metrics, method, n_points, n_samples, log_P_x, log_q_x, log_a_x, phi_x):
    points = []
    for i in trange(n_points):
        log_P_xi, log_q_xi, log_a_xi, phis_xi = bootstrap_resample(method, 
                i, n_samples, log_P_x, log_q_x, log_a_x, phi_x)
        point = get_metrics(log_P_xi, log_q_xi, log_a_xi, phis_xi)
        points.append(point)
    return pd.DataFrame(points)

def get_metrics_estimates_proposal(log_P_xi, log_q_xi, log_a_xi, phis_xi):
    Z = np.mean(np.exp(log_P_xi - log_q_xi))
    log_Z = np.log(Z)
    log_p_xi = log_P_xi - log_Z
    tvd = 0.5 * np.mean(np.abs(1. - np.exp(log_p_xi - log_q_xi)))
    kl_p = np.mean(np.where(np.isneginf(log_P_xi), 0., np.exp(log_p_xi - log_q_xi) * (log_p_xi - log_q_xi)))
    if log_a_xi is not None:
        kl_a = np.mean(log_q_xi - log_a_xi)
    else:
        kl_a = None
    estimates = {
            'beta': 0,
            'ar': 1,
            'tvd': tvd,
            'kl_p': kl_p,
            'kl_a': kl_a,
            'Z': Z,
            'log_Z': log_Z
            }
    for name in phis_xi:
        phi_xi = phis_xi[name]
        cs = np.mean(phi_xi)
        estimates[f"constraint-{name}"] = cs
        #estimate EBM moments
        cs = np.mean(np.exp(log_p_xi - log_q_xi) * phi_xi)
        estimates[f"p-constraint-{name}"] = cs
    return estimates

def get_metrics_estimates_naive_filter(log_P_xi, log_q_xi, log_a_xi, phis_xi):
    Z = np.mean(np.exp(log_P_xi - log_q_xi))
    log_Z = np.log(Z)
    log_p_xi = log_P_xi - np.log(Z) 
    ar = np.mean(phis_xi.iloc[:,0])
    log_NF = np.where(np.isneginf(log_P_xi), -np.infty, log_q_xi)
    Z_NF = np.mean(np.exp(log_NF - log_q_xi))
    log_nf = log_NF - np.log(Z_NF)
    tvd = 0.5 * np.mean(np.where(np.isneginf(log_P_xi), 0., np.abs(np.exp(log_nf - log_q_xi) - np.exp(log_p_xi - log_q_xi))))
    kl_p = np.mean(np.where(np.isneginf(log_P_xi), 0., np.exp(log_p_xi - log_q_xi) * (log_p_xi - log_nf)))
    if log_a_xi is not None:
        kl_a = np.mean(np.where(np.isneginf(log_NF), 0., np.exp(log_nf - log_q_xi) * (log_nf - log_a_xi)))
    else:
        kl_a = None
    estimates = {
            'beta': 0,
            'ar': ar,
            'tvd': tvd,
            'kl_p': kl_p,
            'kl_a': kl_a,
            'Z': Z,
            'log_Z': log_Z
            }
    for name in phis_xi:
        estimates[f"constraint-{name}"] = 1.
    return estimates


def safer_log(x):
    if x > 0:
        return np.log(x)
    else:
        return float("-inf")
main()
