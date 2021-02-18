# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import argparse
import pickle
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ebm_scores', type=Path)
    ap.add_argument('-M', '--upper', type=float)
    args = ap.parse_args()


    #print('Loading EBM scores...')
    log_P_x = pickle.load(open(args.ebm_scores, 'rb'))
    #print('Loading proposal scores...')
    log_a_x = pickle.load(open(args.ebm_scores.with_suffix('').with_suffix('.base.pkl'), 'rb'))

    if args.upper is None:
        upper = np.exp(np.max(log_P_x - log_a_x))
    else:
        upper = args.upper
    print(f'upper bound is {upper}')

    betas_x = np.exp(log_P_x - log_a_x)

    Z = betas_x.mean()
    Z_std = upper**2 / (4 * len(betas_x))
    print(f'{Z:.2g} pm {Z_std:.2g}')
    print(f'(using {len(betas_x)} samples)')
    
main()
