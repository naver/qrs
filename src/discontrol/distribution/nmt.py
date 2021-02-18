# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Model, GPT2PreTrainedModel
from transformers import MarianMTModel, MarianTokenizer
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from discontrol.scorer import BaseScorer
from discontrol.misc import TorchArray, ListArray, torch_array
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Identity, CrossEntropyLoss
import torch.nn as nn
import sys

from discontrol.misc import torch_array

class FBRoundTripNMT(BaseScorer):

    def __init__(self, fwd_model_name, bwd_model_name, orig, batch_size=1, temperature=1., topk=0):
        self.fwd_tokenizer = FSMTTokenizer.from_pretrained(fwd_model_name)
        self.fwd_model = FSMTForConditionalGeneration.from_pretrained(fwd_model_name)
        self.fwd_model.eval()
        self.fwd_model.requires_grad = False
        if bwd_model_name is None:
            self.bwd_model = self.fwd_model 
            self.bwd_tokenizer = self.fwd_tokenizer
        else:
            self.bwd_tokenizer = FSMTTokenizer.from_pretrained(bwd_model_name)
            self.bwd_model = FSMTForConditionalGeneration.from_pretrained(bwd_model_name)
        self.bwd_model.eval()
        self.bwd_model.requires_grad = False
        self.device = torch.device('cpu')
        self.batch_size = batch_size
        self.temperature = temperature
        self.orig = orig
        self.log_scores = {}
        self.y_beam_ids = None
        self.topk = topk

    def to(self, device):
        self.fwd_model = self.fwd_model.to(device)
        self.bwd_model = self.bwd_model.to(device)
        self.device = device
        return self

    def sample_and_score(self, return_beam=False):
        """
        returns:
            samples: ListArray of samples (strings)
            scores: log probabilities of the sequences
        """
        x = self.orig

        # Translate using the forward model and beam search.
        if self.y_beam_ids is None:
            x_ids = self.fwd_tokenizer(x, return_tensors="pt").to(self.device)
            y_beam_ids = self.fwd_model.generate(**x_ids, do_sample=False)
            y_beam = [self.fwd_tokenizer.decode(y_ids, skip_special_tokens=True) for y_ids in y_beam_ids]
            
            # Tokenize for the backward model
            y_beam_ids = self.bwd_tokenizer(y_beam, padding=True, return_tensors="pt").to(self.device)
            self.y_beam_ids = y_beam_ids
        else:
            y_beam_ids = self.y_beam_ids

        # Sample from the backward model.
        x_prime_ids = self.bwd_model.generate(**y_beam_ids, do_sample=True, top_k=self.topk, output_scores=True, num_beams=1,
                                              num_return_sequences=self.batch_size, temperature=self.temperature, 
                                              return_dict_in_generate=True)
        x_prime = [self.bwd_tokenizer.decode(x_ids, skip_special_tokens=True) for x_ids in x_prime_ids.sequences]

        properties = {} # {'ids': TorchArray(x_prime_ids.sequences)}
        samples = ListArray(x_prime, properties)

        # Compute log probabilties
        log_probs = torch.stack(x_prime_ids.scores, dim=1).log_softmax(-1) # [num_return_sequences, max_length, vocab_size]
        shifted_out_ids = x_prime_ids.sequences[:, 1:].unsqueeze(2)
        log_probs = log_probs.gather(2, shifted_out_ids).squeeze(2)
        mask = (x_prime_ids.sequences[:, 1:] != self.bwd_tokenizer.pad_token_id)
        log_probs = torch.where(mask, log_probs, log_probs.new_zeros(log_probs.size()))
        log_probs = log_probs.sum(-1) # compute sequence probs

        if return_beam:
            return samples, log_probs, y_beam[0]
        else:
            return samples, log_probs

    def sample(self):
        samples, scores = self.sample_and_score()
        self.log_scores.update({sample: score for sample, score in zip(samples, scores)})
        return samples

    def score(self, x):
        return self.log_score(x).exp()

    def log_score(self, x): 
        try:
            return torch_array([self.log_scores[xi] for xi in x])
        except:
            raise NotImplementedError
