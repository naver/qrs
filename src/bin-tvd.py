# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

def main():
    path = Path("<output-dir>")
    x = pickle.load(open(path / "1M/amazing/direct-amazing.pkl", 'rb'))
    log_q_x = pickle.load(open(path / "1M/amazing/direct-amazing.proposal.pkl", 'rb'))
    log_P_x = pickle.load(open(path / "1M/amazing/direct-amazing.ebm.pkl", 'rb'))

    binning = XBinning()
    p_bins = project_p(binning, x, log_P_x, log_q_x)
    for sampler in tqdm(["rwmhreset-0.001-amazing-0.1","rwmh-0.001-amazing-0.1", "qrs-0.001", "imh-0.001", "imhreset-0.001"]):
        tvds = []
        for i in range(2, 11):
            y = pickle.load(open(path / f"10k/{i}/amazing/{sampler}.pkl", 'rb'))
            y_bins = to_bins(y, binning)
            b_y = bins_to_probs(y_bins)
            tvd = np.abs(p_bins - b_y).sum() / 2
            tvds.append(tvd)
        tvds = np.array(tvds)
        print(f'{sampler} TVD >= {tvds.mean():.2f} pm {tvds.std():.2f}')


def project_p(binning, x, log_P_x, log_q_x):
    Z = np.exp(log_P_x - log_q_x).mean()
    log_p_x = log_P_x - np.log(Z)
    bins = to_bins(x, binning)
    if (bins == bins[0]).all():
        raise RuntimeError("Bins don't partition the space")
    p_bins = np.zeros(binning.n_bins())
    for b in range(len(p_bins)):
        beta = np.exp(log_p_x - log_q_x)
        p_bins[b] = (beta*(bins == b)).mean()
    p_bins /= np.sum(p_bins)
    print(p_bins)
    return p_bins

def to_bins(x, binning):
    bins = np.zeros(len(x))
    for i, x_i in enumerate(tqdm(x, desc="Projecting into bins")):
        bins[i] = binning(x_i)
    return bins

def bins_to_probs(y):
    bins = int(y.max()) + 1
    p = np.zeros(bins)
    for b in range(bins):
        p[b] = np.sum(y == b)/len(y)
    print(p)
    return p

class XBinning(object):
    def n_bins(self):
        return 2

    def __call__(self, x):
        return x.count('\n') > 0

main()
