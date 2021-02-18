# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import argparse
import pickle
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('samples_fn', type=Path)
    ap.add_argument('-n', '--n-splits', default=10, type=int)

    args = ap.parse_args()


    print(args.samples_fn.stem)
    #print('Loading EBM scores...')
    log_P_x = pickle.load(open(args.samples_fn.with_suffix('.ebm.pkl'), 'rb'))
    #print('Loading proposal scores...')
    log_q_x = pickle.load(open(args.samples_fn.with_suffix('.proposal.pkl'), 'rb'))

    betas_x = np.exp(log_P_x - log_q_x)

    N = len(log_P_x) // args.n_splits
    betas_x = betas_x.reshape((args.n_splits, N))
    Z = betas_x.mean(1)
    print(f'{Z.mean():.2g} pm {Z.std():.2g}')
    
main()
