# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import importlib
import logging
logger = logging.getLogger(__name__)


def build_sampler_from_config(config, cuda=True):
    p = build_distribution(config['base'])
    if cuda:
        p = p.to('cuda')
    if 'proposal' in config:
        q = build_distribution(config['proposal'])
        if cuda:
            q = q.to('cuda')
    else:
        q = p
    P = build_ebm_from_config(config, p=p)
    if 'sampler' in config:
        sampler = build_sampler(config['sampler'], P, q)
    else:
        if 'filters' in config:
            logger.warn("You need to provide a sampler class if you want to guarantee your filters.")
        sampler = q
    return sampler

def build_distribution(cfg):
    return build_class_and_args('distribution', cfg)

def build_ebm_from_config(config, p=None):
    if p is None:
        p = build_distribution(config['base'])
    if 'filters' in config:
        return apply_filters(p, build_classes_and_args('scorer', config['filters']))
    else:
        return p

def build_sampler(sampler_cfg, P, q):
    return build_class_and_args("sampler", sampler_cfg, ebm=P, proposal=q)

def build_classes_and_args(namespace, class_configs):
    return [build_class_and_args(namespace, cfg) for cfg in class_configs]

def build_class_and_args(namespace, cfg, *args, **kwargs):
    cfg_args = build_args(namespace, cfg.get('args', {}))
    if isinstance(cfg_args, list):
        args = list(args) + cfg_args
    elif isinstance(cfg_args, dict):
        cfg_args.update(kwargs)
        kwargs = cfg_args
    return build_class(namespace + '.' + cfg['class'], *args, **kwargs)

def build_args(namespace, args):
    if isinstance(args, list):
        return [build_class_and_args(namespace, arg) if is_class_spec(arg) 
                else build_args(namespace, arg)
                for arg in args]
    elif isinstance(args, dict):
        return {k: build_class_and_args(namespace, arg) if is_class_spec(arg)
                else build_args(namespace, arg)
                for k, arg in args.items()}
    else:
        return args


def is_class_spec(arg):
    return isinstance(arg, dict) and 'class' in arg

def build_class(cls_path, *args, **kwargs):
    cls = get_class(cls_path)
    return cls(*args, **kwargs)

def get_class(cls_path, top_module='discontrol'):
    if top_module:
        cls_path = f'{top_module}.' + cls_path
    module_name, cls_name = cls_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls


def apply_filters(p, filters):
    P = p
    for fltr in filters:
        P = P * fltr
    return P

