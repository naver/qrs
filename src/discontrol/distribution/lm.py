# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Model, GPT2PreTrainedModel
from discontrol.scorer import BaseScorer
from discontrol.misc import TorchArray, ListArray, torch_array
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Identity, CrossEntropyLoss
import torch.nn as nn
import sys

class BaseAutoregressiveLM(BaseScorer):

    def __init__(self, model, tokenizer, prompt=None, include_prompt=False,
            _prompt_acceptance_rate=None,
            max_length=41,
            min_length=41,
            **sampling_config):
        """Configuration parameters (same as those in the transfomers library):
        see https://huggingface.co/transformers/main_classes/model.html#transformers.generation_utils.GenerationMixin.generate
        """
        self.model = model
        self.model.eval()  # better be safe than sorry
        self.model.requires_grad = False
        self.tokenizer = tokenizer
        default_config = {"do_sample" : True, 
                "top_k": 0}
        default_config.update(sampling_config)
        self.sampling_config = default_config
        self.device = 'cpu'

        self.include_prompt = include_prompt
        self.prompt_length = 0
        self.input_ids = None
        self.prompt = prompt
        self._prompt_acceptance_rate = _prompt_acceptance_rate
        self._prompt_total_samples = 0
        self._prompt_accepted_samples = 0
        if prompt:
            sys.stdout.write(f'Loaded prompt {repr(prompt)}\n')
            sys.stdout.flush()
            batch_size = self.sampling_config["num_return_sequences"] if "num_return_sequences" in self.sampling_config else 1
            self.input_ids = torch.cat([torch.tensor([[tokenizer.eos_token_id]]), self.tokenizer.encode(prompt, return_tensors="pt")], 1).repeat(batch_size, 1).to(self.device)
            self.prompt_length = self.input_ids.size()[-1]
        self.max_length = max_length
        self.min_length = max_length

    def to(self, device):
        self.model = self.model.to(device)
        if self.input_ids is not None:
            self.input_ids = self.input_ids.to(device)
        self.device = device
        return self

    def sample(self):
        out_ids = self.model.generate(**self.sampling_config,
                input_ids=self.input_ids,
                max_length=self.max_length + self.prompt_length,
                min_length=self.min_length + self.prompt_length,
                bos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id).to('cpu')
        if out_ids.dim() == 1:
            out_ids = out_ids.unsqueeze(0)
        #out_ids = torch.cat([out_ids[:, 0].unsqueeze(1), out_ids[:, self.prompt_length:]], 1)
        properties = {}
        sequences = self.tokenizer.batch_decode(out_ids)
        if self.prompt:
            self._prompt_total_samples += len(sequences)
            out_ids, sequences = self.keep_deterministic(out_ids, sequences)
            if not self.include_prompt:
                sequences = self.remove_prompt(sequences)
            self._prompt_accepted_samples += len(sequences)
        else:
            properties['ids'] = TorchArray(out_ids)
        samples = ListArray(sequences, properties)
        return samples

    def keep_deterministic(self, out_ids, sequences):
        kept_ids, kept_sequences = [], []
        for ids, s in zip(out_ids, sequences):
            ids2 = self.tokenizer.encode(s, return_tensors='pt')[0]
            if ids.shape == ids2.shape and (ids == ids2).all():
                kept_ids.append(ids)
                kept_sequences.append(s)
        return torch.stack(kept_ids), kept_sequences

    def remove_prompt(self, sequences):
        new_sequences = []
        prompt = self.tokenizer.eos_token + self.prompt + " "
        for s in sequences:
            if s.startswith(prompt):
                s = self.tokenizer.eos_token + s[len(prompt):]
                new_sequences.append(s)
            else:
                pass
                #print(f'unmatched prompt: {s}')
        return new_sequences

    def reinsert_prompt(self, sequences):
        new_sequences = []
        prompt = self.tokenizer.eos_token + self.prompt + " "
        for s in sequences:
            if s.startswith(self.tokenizer.eos_token):
                s = prompt + s[len(self.tokenizer.eos_token):]
                new_sequences.append(s)
            else:
                pass
                #print(f'unrecognized sample format: {s}')
        return new_sequences

    def score(self, samples):
        return self.log_score(samples).exp()

    def log_score(self, samples):
        drop_logits = 0
        if not samples.has_property('ids') or (self.prompt and not self.include_prompt):
            if self.prompt:
                if not self.include_prompt:
                    # if self.include_prompt is true, samples will already contain the prompt.
                    samples = self.reinsert_prompt(samples)
                drop_logits = self.prompt_length - 1
            sample_ids = [self.tokenizer.encode(s, return_tensors='pt').to(self.device) 
                           for s in samples]
        else:
            sample_ids = samples.get_property('ids').data.to(self.device)

        log_probs = self.get_logprobs(sample_ids, drop_first_logits=drop_logits)

        if self.prompt and not self.include_prompt:
            log_probs -= np.log(self._prompt_acceptance_rate)

        return log_probs

    def get_logprobs(self, sample_ids, drop_first_logits=0): 
        try:
            if isinstance(sample_ids, list):
                sample_ids = torch.cat(sample_ids, 0)
        except:
            return self.get_logprobs_sequential(sample_ids, drop_first_logits=drop_first_logits)
        return self.get_logprobs_batch(sample_ids, drop_first_logits=drop_first_logits)

    def get_logprobs_batch(self, sample_ids, drop_first_logits=0):
        logits = F.log_softmax(self.model(sample_ids, return_dict=True).logits, dim=2)
        shifted_ids = sample_ids[:,1:]
        log_probs = TorchArray(logits.gather(2, shifted_ids.unsqueeze(2)).squeeze(2)[:, drop_first_logits:].sum(1)).to('cpu')
        return log_probs

    def get_logprobs_sequential(self, sample_ids, drop_first_logits=0):
        log_probs = []
        for sids in sample_ids:
            logits = F.log_softmax(self.model(sids, return_dict=True).logits, dim=2)
            shifted_ids = sids[:,1+drop_first_logits:].unsqueeze(2)
            log_probs_i = logits.gather(2, shifted_ids).squeeze(2).sum().to('cpu')
            log_probs.append(log_probs_i)
        log_probs = TorchArray(torch_array(log_probs))
        return log_probs

    def load_checkpoint(self, fn):
        state = torch.load(fn)
        self.model.load_state_dict(state['model_state'])

    def get_meta(self):
        return {'prompt_acceptance_rate': 
                self._prompt_accepted_samples / self._prompt_total_samples}

class HFAutoregressiveLM(BaseAutoregressiveLM):

    def __init__(self, model_name, **sampling_config):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        super(HFAutoregressiveLM, self).__init__(model, tokenizer, 
                **sampling_config)


class GDCAutoregressiveLM(BaseAutoregressiveLM):
    def __init__(self, model_name, checkpoint_fn, **sampling_config):
        state = torch.load(checkpoint_fn, map_location='cuda:0')
        state_dict = index_state_dict(state['model_state'], 'transformer')
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                state_dict=state_dict)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        super(GDCAutoregressiveLM, self).__init__(model, tokenizer, 
                **sampling_config)

def index_state_dict(state_dict, *keys):
    return {k[k.find('.')+1:]: v for k,v in state_dict.items()
        if any([k.startswith(f'{key}.') for key in keys])}

