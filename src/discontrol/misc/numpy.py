# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from discontrol.misc.base_array import BaseArray
import numpy as np

class NumpyArray(BaseArray):
    def __init__(self, data):
        super(NumpyArray, self).__init__(data)

    def new(self, ot):
        if isinstance(ot, NumpyArray):
            return NumpyArray(np.copy(ot.data))
        else:
            return NumpyArray(ot)

    def clone(self):
        return self.new(self)

    def cast(self, ot):
        if isinstance(ot, NumpyArray):
            return ot
        else:
            return self.new(ot)

    def apply_mask(self, mask):
        return self.new(np.multiply(self, mask))

    def filter(self, mask):
        return self.new(self.data[mask])

    def concat(self, other):
        return self.new(np.concatenate((self.data, other.data)))

    def concat_to(self, other):
        if other:
            return self.new(np.concatenate((other.data, self.data)))
        else:
            return self.new(self.data)

    def clip(self, minval, maxval):
        return self.new(np.clip(self.data, min=minval, max=maxval))

    def new_uniform(self, lower, upper):
        return NumpyArray(np.random.uniform(0, 1, len(self.data)))

    def maxval(self):
        return self.data.max()

    def item(self):
        if len(self.data) != 1: raise ValueError("Only 1-dimensional arrays can be converted to Python scalars.")
        return self.data[0]

    @classmethod
    def cat(self, arrs):
        return NumpyArray(np.concatenate([arr.data for arr in arrs]))

    def log(self):
        return self.new(np.log(self.data))

    def exp(self):
        return self.new(np.exp(self.data))

    def sortval(self):
        return self.new(np.sort(self.data))

def np_array(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    return NumpyArray(data)
