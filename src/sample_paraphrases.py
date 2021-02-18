# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from transformers import MarianMTModel, MarianTokenizer

from discontrol.distribution.nmt import FBRoundTripNMT
from discontrol.distribution.lm import HFAutoregressiveLM
from discontrol.scorer import KeywordsFeature, ProductScorer, ParaphraseFeature

from discontrol.misc import ListArray
from discontrol.sampler import QuasiRejectionSampler, QuasiRejectionSamplerFixedBeta

from pathlib import Path
from tqdm import tqdm

import numpy as np 
import argparse
import torch

def main(seq_idx, cutoff):

    orig = [
        'How is the two wheeler insurance from Bharti Axa insurance?',
        'Are there Doctor Who references in the Muse song "Knights of Cydonia"?',
        'In French, how do you say "cool"?'
    ][seq_idx]
    print(f"input sequence: {orig}")

    temperature = 1.0
    topk = 30
    fwd_model_name = "facebook/wmt19-en-de"
    bwd_model_name = "facebook/wmt19-de-en"
    batch_size = 50
    device = torch.device("cuda")
    num_samples = 1_000_000

    q = FBRoundTripNMT(fwd_model_name, bwd_model_name, orig, batch_size=batch_size, temperature=temperature, topk=topk)
    a = HFAutoregressiveLM("gpt2")
    a.to(device)
    q.to(device)

    partial_matches = False
    minimal_match = 0.5
    keyword_feature = KeywordsFeature(orig, partial_matches=False)

    cutoff_p = cutoff / 100.
    print(f"cutoff = {cutoff_p:.3f}")
    model_name = "all-mpnet-base-v2"
    paraphrase_feature = ParaphraseFeature(orig, model_name=model_name, cutoff=cutoff_p, device=device)
    
    out_dir = Path(f"<out-dir>/paraphrasing-{cutoff}/{seq_idx}")
    out_dir.mkdir(exist_ok=True, parents=True)
    base_filename = "round-trip-nmt"
    log_q = np.array([])
    log_a = np.array([])
    log_P_unsup = np.array([])
    log_P_sup = np.array([])
    feature_sup = np.array([])
    feature_unsup = np.array([])
    samples = []
    pbar = tqdm(total=num_samples)
    while len(samples) < num_samples:
        proposals_i, log_qi = q.sample_and_score()
        log_q = np.concatenate([log_q, log_qi.cpu()])
        samples.extend(proposals_i)
       
        feature_sup_i = paraphrase_feature.score(proposals_i)
        feature_unsup_i = keyword_feature.score(proposals_i)
        feature_sup = np.concatenate([feature_sup, feature_sup_i])
        feature_unsup = np.concatenate([feature_unsup, feature_unsup_i])

        input_P = ListArray([f"<|endoftext|>{seq}" for seq in proposals_i])
        log_a_i = a.log_score(input_P).cpu().numpy()

        log_P_sup_i = log_a_i + np.log(feature_sup_i.data)
        log_P_unsup_i = log_a_i + np.log(feature_unsup_i.data)
        log_P_sup = np.concatenate([log_P_sup, log_P_sup_i])
        log_P_unsup = np.concatenate([log_P_unsup, log_P_unsup_i])
        log_a = np.concatenate([log_a, log_a_i])
        pbar.update(batch_size)
    pbar.close()

    np.save(out_dir / f"{base_filename}.npy", samples)
    np.save(out_dir / f"{base_filename}.proposal.npy", log_q)
    np.save(out_dir / f"{base_filename}.base.npy", log_a)
    np.save(out_dir / f"{base_filename}.ebm-sup.npy", log_P_sup)
    np.save(out_dir / f"{base_filename}.ebm-unsup.npy", log_P_unsup)
    np.save(out_dir / f"{base_filename}.feature-unsup.npy", feature_unsup)
    np.save(out_dir / f"{base_filename}.feature-sup.npy", feature_sup)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_idx', type=int)
    parser.add_argument('cutoff', type=int)
    args = parser.parse_args()
    main(args.seq_idx, args.cutoff)
