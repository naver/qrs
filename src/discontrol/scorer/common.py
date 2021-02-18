# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from discontrol.misc import np_array

def multiply(*xs):
    ret = None
    for x in xs:
        if ret is not None:
            ret *= ret.cast(x)
        else:
            ret = x.clone()
    return ret

def is_atomic(xs):
    return any(isinstance(xs, t) for t in (str, int, float, bool))

def broadcast(predicate):
    def broadcasted_predicate(xs):
        if is_atomic(xs):
            raise TypeError(f"{xs} should be of a list/array-like type")
        return np_array([predicate(x) for x in xs])
    return broadcasted_predicate

def do_batched(func, batch_size, *args):
    all_ret = []
    for i in range(len(args[0])//batch_size + 1):
        batch_args = [arg[i*batch_size:(i+1)*batch_size] for arg in args]
        if batch_args[0]:
            all_ret.append(func(*batch_args))
    return type(all_ret[0]).cat(all_ret)
