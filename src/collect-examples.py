# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import streamlit as st
from pathlib import Path
import pickle
import pandas as pd
import re
from pylatexenc.latexencode import unicode_to_latex
pd.set_option("max_colwidth", 0)

def main():
    path = Path("<output-dir>/")
    print_samples(path, 'QRS samples at $10^{-3}$ acceptance rate', "qrs-0.001")
    print_samples(path, 'RWMH samples at $10^{-3}$ acceptance rate', "rwmh-0.001-amazing-0.1")
    print_samples(path, 'RWMH-R samples at $10^{-3}$ acceptance rate', "rwmhreset-0.001-amazing-0.1")
    print_samples(path, 'IMH samples at $10^{-3}$ acceptance rate', "imh-0.001")
    print_samples(path, 'IMH-R samples at $10^{-3}$ acceptance rate', "imhreset-0.001")



def print_samples(path, title, sampler):
    qrs_samples = pd.DataFrame({title: collect_samples(path / f"10k/1/amazing/{sampler}.pkl")})
    st.code(qrs_samples.to_latex(escape=False, index=False, column_format=r"p{\linewidth}"))

def collect_samples(path):
    xs = pickle.load(open(path, 'rb'))
    return [fmt(x) for x in xs[-10:]]

def fmt(x):
    return r"\textbullet\ " + r"\tt " + tex_escape(x[len("<|endoftext|>"):]).replace("amazing", r"\textbf{amazing}")

def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '\n': r'\keys{\return}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key = lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)

main()
