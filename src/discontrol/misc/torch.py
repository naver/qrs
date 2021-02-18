# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from discontrol.misc.base_array import BaseArray, override
import torch

class TorchArray(BaseArray):
    def __init__(self, data):
        super(TorchArray, self).__init__(data)

    def new(self, ot):
        if isinstance(ot, torch.Tensor):
            return TorchArray(ot.data.clone().detach())
        else:
            return TorchArray(torch.tensor(ot, device=self.data.device))

    def cast(self, ot):
        if isinstance(ot, TorchArray):
            return ot.to(self.data.device)
        else:
            return self.new(ot)

    def apply_mask(self, mask):
        return self.new(torch.mul(self.data, mask.data))

    def filter(self, mask):
        return self.new(self.data[mask.data])

    def concat(self, other):
        return self.new(torch.cat((self.data, other.data)))

    def concat_to(self, other):
        if other:
            return self.new(torch.cat((other.data, self.data)))
        else:
            return self.new(self.data)

    def clip(self, minval, maxval):
        return self.new(torch.clip(self.data, min=minval, max=maxval))

    def new_uniform(self, lower, upper):
        return self.new(torch.rand(self.size(), device=self.data.device) * (upper - lower) + lower)

    def maxval(self):
        return self.data.max().item()

    def sortval(self):
        return self.new(self.data.sort().values)

    def any(self):
        return self.data.any().item()

    def item(self):
        return self.data.item()

    @override
    def __getitem__(self, key):
        ret = self.data[key]
        if ret.dim() == 0:
            return ret.item()
        else:
            return self.new(ret)

    @classmethod
    def cat(self, arrs):
        return TorchArray(torch.cat([arr.data for arr in arrs]))

def torch_array(data):
    if isinstance(data, TorchArray):
        data = data.data
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    return TorchArray(data)
