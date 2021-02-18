# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license


class ListArray(list):
    def __init__(self, data, aux_data={}):
        super(ListArray, self).__init__(data)
        self.aux_data = aux_data

    def __getitem__(self, key):
        val = super(ListArray, self).__getitem__(key)
        if isinstance(val, list):
            return self.new(val, 
                    {k:v[key] for k,v in self.aux_data.items()})
        else:
            return val

    def has_property(self, name):
        return name in self.aux_data

    def get_property(self, name):
        return self.aux_data[name]

    def set_property(self, name, value):
        self.aux_data[name] = value

    @classmethod
    def new(self, ot, aux_data={}):
        if isinstance(ot, ListArray):
            if aux_data:
                return ListArray(ot, aux_data)
            else:
                return ListArray(ot, ot.aux_data)
        else:
            return ListArray(ot, aux_data)

    def cast(self, ot):
        if isinstance(ot, ListArray):
            return ot
        else:
            return self.new(ot, {})

    def filter(self, mask):
        aux_data = {k: v.filter(mask) for k,v in self.aux_data.items()}
        return self.new([x for x,m in zip(self, mask) if m], aux_data)

    def concat(self, other):
        new_aux_data = {}
        for k in self.aux_data:
            new_aux_data[k] = self.aux_data[k].concat(other.aux_data[k])
        self.extend(other)
        return self.new(self, new_aux_data)

    def concat_to(self, other):
        if other is None:
            return self.new(self, self.aux_data)
        else:
            return other.concat(self)

    # def __getitem__(self, item):
    #     return self.new(self.data[item],
    #             {k:v[item] for k,v in self.aux_data.items()})

    def __getstate__(self):
        return (list(self), self.aux_data)

    def __setstate__(self, state):
        self[:], self.aux_data = state


def list_array(data):
    if not isinstance(data, list):
        data = list(data)
    return ListArray(data)
