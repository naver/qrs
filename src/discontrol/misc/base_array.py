# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

def override(method):
    '''Allows to override special methods'''
    method.overriden = True
    return method

class BaseArray(object):
    '''based on https://code.activestate.com/recipes/496741-object-proxying/'''
    __slots__ = ["_obj", "__weakref__"]
    def __init__(self, obj):
        object.__setattr__(self, "_obj", obj)

    @property
    def data(self):
        return object.__getattribute__(self, "_obj")
    
    #
    # proxying (special cases)
    #
    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            attr = getattr(object.__getattribute__(self, "_obj"), name)
            if callable(attr):
                return wrap_bound_method(name, self, attr)
            else:
                return attr
    
    def __nonzero__(self):
        return bool(object.__getattribute__(self, "_obj"))
    def __str__(self):
        return str(object.__getattribute__(self, "_obj"))
    def __repr__(self):
        return repr(object.__getattribute__(self, "_obj"))
    
    #
    # factories
    #
    _special_names = [
        '__abs__', '__add__', '__and__', '__call__', '__cmp__', '__coerce__', 
        '__contains__', '__delitem__', '__delslice__', '__div__', '__divmod__', 
        '__eq__', '__float__', '__floordiv__', '__ge__', '__getitem__', 
        '__getslice__', '__gt__', '__hash__', '__hex__', '__iadd__', '__iand__',
        '__idiv__', '__idivmod__', '__ifloordiv__', '__ilshift__', '__imod__', 
        '__imul__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', 
        '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', 
        '__long__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', 
        '__neg__', '__oct__', '__or__', '__pos__', '__pow__', '__radd__', 
        '__rand__', '__rdiv__', '__rdivmod__', '__reduce__', '__reduce_ex__', 
        '__repr__', '__reversed__', '__rfloorfiv__', '__rlshift__', '__rmod__', 
        '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', 
        '__rtruediv__', '__rxor__', '__setitem__', '__setslice__', '__sub__', 
        '__truediv__', '__xor__', 'next',
    ]
    
    @classmethod
    def _create_class_proxy(cls, theclass):
        """creates a proxy for the given class, wrapping and unwrapping
        function arguments and return rtpes"""
        namespace = {}
        for name in cls._special_names:
            if hasattr(theclass, name) and \
                    not (hasattr(cls, name) and 
                            hasattr(getattr(cls, name), 'overriden')):
                namespace[name] = wrap_cls_method(name)
        return type("%s(%s)" % (cls.__name__, theclass.__name__), (cls,), namespace)
    
    def __new__(cls, obj, *args, **kwargs):
        """
        creates an proxy instance referencing `obj`. (obj, *args, **kwargs) are
        passed to this class' __init__, so deriving classes can define an 
        __init__ method of their own.
        note: _class_proxy_cache is unique per deriving class (each deriving
        class must hold its own cache)
        """
        try:
            cache = cls.__dict__["_class_proxy_cache"]
        except KeyError:
            cls._class_proxy_cache = cache = {}
        try:
            theclass = cache[obj.__class__]
        except KeyError:
            cache[obj.__class__] = theclass = cls._create_class_proxy(obj.__class__)
        ins = object.__new__(theclass)
        theclass.__init__(ins, obj, *args, **kwargs)
        return ins

def wrap_bound_method(name, self, orig_method):
    def method(*args, **kw):
        args = [arg.data if type(arg) == type(self) else arg for arg in args]
        ret = orig_method(*args, **kw)
        if type(ret) == type(object.__getattribute__(self, "_obj")):
            return object.__getattribute__(self, "new")(ret)
        else:
            return ret
    return method

def wrap_cls_method(name):
    def method(self, *args, **kw):
        args = [arg.data if type(arg) == type(self) else arg for arg in args]
        f_method = getattr(object.__getattribute__(self, "_obj"), name)
        ret = f_method(*args, **kw)
        if type(ret) == type(object.__getattribute__(self, "_obj")):
            return object.__getattribute__(self, "new")(ret)
        else:
            return ret
    return method
