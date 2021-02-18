# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from . import BaseSampler, AccumulatorSampler
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch
import numpy as np

class LocalProposal(BaseSampler):

    def __init__(self, mix_with_delta=None, gamma=0.1):
        """
        :param mix_with_delta: mix insert / replace probabilities with a dirac delta on a particular word.
        :param gamma: between 0 and 1, strength of dirac delta
        """
        self.action_probs = [1./3, 1./3, 1./3] # delete, replace, insert
        self.action_probs_empty = [0., 0., 1.]
        self.action_probs_max_length = [1./2, 1./2, 0]
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
        p_a = self.action_probs
        if len(tok_x) == 0: p_a = self.action_probs_empty
        if len(tok_x) >= 500: p_a = self.action_probs_max_length
        action = np.random.choice(3, p=p_a)
        if action == 0:
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
