# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import os
import sys
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import multiprocessing

import sacrebleu
import nltk

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from pathlib import Path
from cycler import cycler
from transformers import AutoTokenizer

from discontrol.sampler import QuasiRejectionSampler, IMHResetSampler
from discontrol.distribution.samplerepo import SampleRepository
from discontrol.scorer.common import broadcast
from discontrol.scorer import SingleWordFeature, GenderFeature, MultiWordFeature, ExponentialScorer
from discontrol.pipeline import build_sampler_from_config
from discontrol.metrics import Distinct_N, SelfBLEU
from discontrol.misc import NumpyArray, list_array, torch_array
from discontrol.sampler import BaseSampler

from qrs_tvd import  get_tvd_estimates, add_bounds, project_into_curve_x

# Plot settings
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
         '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
custom_cycler = cycler(color=colors)
plt.rc('axes', prop_cycle=custom_cycler)
sns.set_palette(sns.color_palette("colorblind"))
MARKER_SIZE=100

def get_constraints(experiment):
    if experiment == "amazing":
        return [("amazing", 1., SingleWordFeature("amazing"))]
    elif experiment == "female":
        return [("female", 0.5, GenderFeature("female"))]
    elif experiment == "female_science":
        return [("female", 0.5, GenderFeature("female")),
                ("science", 1., MultiWordFeature("resources/wikibio-wordlists/science.txt"))]
    elif experiment == "female_sports":
        return [("female", 0.5, GenderFeature("female")),
                ("sports", 1., MultiWordFeature("resources/wikibio-wordlists/sports.txt"))]

from discontrol.pipeline import build_ebm_from_config, build_distribution
def get_ebm(dist_cfg):
    with open(dist_cfg, 'r') as fdistribution_config:
        distribution_config = yaml.safe_load(fdistribution_config)
    P = build_ebm_from_config(distribution_config)
    return P

def get_proposal(sampler_cfg):
    with open(sampler_cfg, 'r') as fsampler_config:
        sampler_config = yaml.safe_load(fsampler_config)
    proposal = build_distribution(sampler_config['proposal'])
    return proposal

def main():
    st.title("Random Walk Metropolis-Hastings")
    sample_size = 1_000
    experiment = "amazing"
    dist_cfg = f"config/lm-distributions/{experiment}.yml"
    sampler_cfg = f"config/samplers/direct-{experiment}.yml"

    # Load data
    q = get_proposal(sampler_cfg)
    P = get_ebm(dist_cfg)
    constraints = get_constraints(experiment)
    q.batch_size = 1

    seed_fn = lambda: q.sample()[0]
    seed = seed_fn()
    # seed = "<|endoftext|>Alan Mathison Turing (7 June 1954) was an English mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist. He was highly influential in the development of theoretical computer science.".lower()
    # seed = "<|endoftext|>A Ferrari passed by my house today.".lower()
    st.write([seed])

    # x = seed
    # samples = []
    # for _ in range(10):
    #     x, log_p, log_p_inv = q_local.sample_and_score(x)
    #     samples.append(x)
    # st.write([samples[-1]])

    q_local = LocalProposal()

    sampler = RWMHSampler(P, q_local, seed, batch_size=10, keep_every=10, burn_in=100)
    st.write(sampler.sample())

    sampler = RWMHResetSampler(P, q_local, 10, seed_fn, batch_size=10)
    st.write(sampler.sample())

    if experiment == "amazing":
        q_local = LocalProposal(mix_with_delta=experiment)

        sampler = RWMHSampler(P, q_local, seed, batch_size=10, keep_every=10, burn_in=100)
        st.write(sampler.sample())

        sampler = RWMHResetSampler(P, q_local, 10, seed_fn, batch_size=10)
        st.write(sampler.sample())

from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch

class LocalProposal(BaseSampler):

    def __init__(self, mix_with_delta=None, gamma=0.1):
        """
        :param mix_with_delta: mix insert / replace probabilities with a dirac delta on a particular word.
        :param gamma: between 0 and 1, strength of dirac delta
        """
        self.action_probs = [1./3, 1./3, 1./3] # delete, replace, insert
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True)
        self.mask_token_id = self.tokenizer.mask_token_id

        if mix_with_delta:
            word_id = self._tokenize(mix_with_delta)
            if len(word_id) > 1:
                raise Exception("mix_with_delta parameter needs to be set with a word that encodes as a single token.")
            word_id = word_id[0]
            delta_dist = torch.Tensor(np.zeros(self.tokenizer.vocab_size))
            delta_dist[word_id] = 1.
            self.compute_p = lambda masked_x: self._run_bert(masked_x) * (1. - gamma) + delta_dist * gamma
        else:
            self.compute_p = self._run_bert

    def _tokenize(self, x):
        return self.tokenizer.encode(x, add_special_tokens=False)

    def _detokenize(self, x):
        return self.tokenizer.decode(x)

    def sample_and_score(self, x):
        x = self._pre_process(x)
        tok_x = self._tokenize(x)
        action = np.random.choice(3, p=self.action_probs)
        if action == 0 and len(tok_x) != 0:
            x_prime, log_p, log_p_inv = self._delete(tok_x)
            log_p += np.log(self.action_probs[0])
            log_p_inv += np.log(self.action_probs[-1])
        elif action == 1:
            x_prime, log_p, log_p_inv = self._replace(tok_x)
            log_p += np.log(self.action_probs[1])
            log_p_inv += np.log(self.action_probs[1])
        else:
            x_prime, log_p, log_p_inv  = self._insert(tok_x)
            log_p += np.log(self.action_probs[-1])
            log_p_inv += np.log(self.action_probs[0])
        return self._post_process(x_prime), log_p, log_p_inv

    def _run_bert(self, masked_x):
        with torch.no_grad():
            input_ids = self.tokenizer.encode_plus(masked_x, return_tensors="pt")
            mask_index = torch.where(input_ids["input_ids"][0] == self.tokenizer.mask_token_id)
            output = self.bert(**input_ids)
            logits = output.logits
            softmax = F.softmax(logits, dim=-1)
            return softmax[0, mask_index, :]

    def _uniform(self, masked_x):
        p = np.ones(self.tokenizer.vocab_size, dtype=np.float32)
        p /= self.tokenizer.vocab_size
        return torch.Tensor([p])

    def _replace(self, tok_x):

        # replace a random word with a mask. 
        pos = np.random.choice(len(tok_x))
        masked_x = tok_x[:pos] + [self.mask_token_id] + tok_x[pos+1:]
        masked_x = self._detokenize(masked_x)

        # run BERT
        p = self.compute_p(masked_x).squeeze()

        # sample a word to fill the mask
        word_id = torch.multinomial(p, 1).item()
        x_prime = tok_x[:pos] + [word_id] + tok_x[pos+1:]
        x_prime = self._detokenize(x_prime)

        # log P
        p = p.cpu().numpy()
        log_p = -np.log(len(tok_x))
        log_p += p[word_id]

        # log P inv
        log_p_inv = -np.log(len(tok_x))
        log_p_inv += p[tok_x[pos]]

        return x_prime, log_p, log_p_inv

    def _insert(self, tok_x):

        # insert a MASK at a random position.
        pos = np.random.choice(len(tok_x)+1)
        masked_x = tok_x[:pos] + [self.mask_token_id] + tok_x[pos:]
        masked_x = self._detokenize(masked_x)

        # run BERT
        p = self.compute_p(masked_x).squeeze()

        # sample a word to fill the mask
        word_id = torch.multinomial(p, 1).item()
        x_prime = tok_x[:pos] + [word_id] + tok_x[pos:]
        x_prime = self._detokenize(x_prime)

        # compute p
        p = p.cpu().numpy()
        log_p = -np.log(len(tok_x)+1)
        log_p += np.log(p[word_id])

        # compute inverse p
        log_p_inv = -np.log(len(tok_x)+1)

        return x_prime, log_p, log_p_inv

    def _delete(self, tok_x):
        pos = np.random.choice(len(tok_x))
        x_prime_tok = tok_x[:pos] + tok_x[pos+1:]
        x_prime = self._detokenize(x_prime_tok)
        
        # log p
        log_p = -np.log(len(tok_x))

        # log p inverse (insert deleted token)
        masked_x_inv = tok_x[:pos] + [self.mask_token_id] + tok_x[pos+1:]
        masked_x_inv = self._detokenize(masked_x_inv)
        p = self.compute_p(masked_x_inv).squeeze().cpu().numpy()
        log_p_inv = -np.log(len(masked_x_inv)+1)
        word_id = tok_x[pos]
        log_p_inv += np.log(p[word_id]) 

        return  x_prime, log_p, log_p_inv

    def _pre_process(self, x):
        """
        remove gpt special tokens.
        """
        return x.replace("<|endoftext|>", "")

    def _post_process(self, x):
        """
        add gpt special tokens.
        """
        return f"<|endoftext|>{x}"

class RWMHSampler(BaseSampler):
    
    def __init__(self, ebm, q_local, seed, batch_size=None, keep_every=1, burn_in=0):
        super(RWMHSampler, self).__init__(ebm, q_local)
        self.keep_every = keep_every
        self.burn_in = burn_in
        self.burnt_in = False
        self.batch_size = proposal.batch_size if batch_size is None else batch_size
        self.accepted = 0
        self.total = 0
        self.q_local = q_local

        self.last = seed
        self.logp_last = ebm.log_score(list_array([seed]))[0]

    def sample(self, return_diagnostics=False):
        chain = []
        alphas = []
        accepted = []
        
        # For the first call to sample(.) we need to run for the burn-in time.
        if not self.burnt_in:
            req_length = self.burn_in + self.batch_size * self.keep_every
        else:
            req_length = self.batch_size * self.keep_every

        while len(chain) < req_length:
            x, logq, logq_inv = self.q_local.sample_and_score(self.last)
            logp = self.ebm.log_score(list_array([x]))[0]
            u = np.random.uniform()
            
            alpha = min(1., np.exp((logp - self.logp_last) + (logq_inv - logq))) 
            alphas.append(alpha)

            if u <= alpha:
                chain.append(x)
                self.logp_last = logp
                self.last = x
                accepted.append(1)
            else:
                chain.append(self.last)
                accepted.append(0)
    
        if not self.burnt_in:
            samples = chain[self.burn_in::self.keep_every]
            self.burnt_in = True
        else:
            samples = chain[::self.keep_every]

        self.accepted += sum(accepted)
        self.total += len(accepted)

        if return_diagnostics:
            diagnostics = {
                "chain": chain,
                "alphas": alphas,
                "accepted": accepted
            }
            return samples, diagnostics
        else:
            return  list_array(samples)

class RWMHResetSampler(BaseSampler):

    def __init__(self, ebm, q_local, reset_every, seed_fn, batch_size=None):
        """ 
        Stateless RWMH sampler.
        """
        super(RWMHResetSampler, self).__init__(ebm, q_local)
        self.reset_every = reset_every
        self.q_local = q_local
        self.seed_fn = seed_fn
        self.batch_size = batch_size

    def sample(self):
        samples = []
        last = self.seed_fn()
        logp_last = self.ebm.log_score(list_array([last]))[0]
        chain = [last]
        
        while len(samples) < self.batch_size:
            x, logq, logq_inv = self.q_local.sample_and_score(last)
            logp = self.ebm.log_score(list_array([x]))[0]
            u = np.random.uniform()
            alpha = min(1., np.exp((logp - logp_last) + (logq_inv - logq))) 

            if u <= alpha:
                chain.append(x)
                logp_last = logp
                last = x
            else:
                chain.append(last)

            if len(chain) > self.reset_every:
                samples.append(chain[-1])
                chain = []
                last  = self.seed_fn()
                log_last = self.ebm.log_score(list_array([last]))[0]
    
        return  list_array(samples)

if __name__ == "__main__":
    main()
