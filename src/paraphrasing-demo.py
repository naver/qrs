# qrs 
# Copyright (c) 2022-present NAVER Corp. 
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import streamlit as st

from transformers import MarianMTModel, MarianTokenizer

from discontrol.scorer import KeywordsFeature, ParaphraseFeature, ProductScorer
from discontrol.distribution.nmt import FBRoundTripNMT
from discontrol.distribution.lm import HFAutoregressiveLM
from discontrol.sampler import QuasiRejectionSampler

def main():
    st.title("Unsupervised Paraphrase Generation")

    orig = st.selectbox(label='sequence', options=[
        'Are there Doctor Who references in the Muse song "Knights of Cydonia"?',
        'How are the two wheeler insurance from Bharti Axa insurance?',
        'In French, how do you say "cool"?'
    ])
    temperature = st.slider(max_value=2.0, min_value=0.1, step=0.1, value=1.0, label='sampling temperature q')
    topk = st.slider(max_value=500, min_value=0, step=10, value=50, label='top-k sampling')
    #partial_matches = st.checkbox("partial matching", value=False)
    #minimal_match = st.slider(min_value=0., max_value=0.9, step=0.1, value=0.5, label="minimal fraction for a partial match")
    cutoff = st.slider(max_value=0.99, min_value=0., step=0.01, value=0.90, label='cosine similarity cutoff')
    nsamples = 10

    # Load q and P
    fwd_model_name = "facebook/wmt19-en-de"
    bwd_model_name = "facebook/wmt19-de-en"
    q = FBRoundTripNMT(fwd_model_name, bwd_model_name, orig, batch_size=nsamples, temperature=1., topk=topk)
    a = HFAutoregressiveLM("gpt2")
    #feature = KeywordsFeature(orig, partial_matches=partial_matches, minimal_match=minimal_match)
    feature = ParaphraseFeature(orig, cutoff=cutoff)
    P = ProductScorer(a, feature)

    # Print keywords
    #st.write("#### Keywords and parts")
    #st.write(feature.keywords)
    #st.write(feature.parts)

    # Sample proposals
    proposals, logq, beam = q.sample_and_score(return_beam=True)
    st.write("#### Forward translation")
    st.write(beam)
    st.write("#### Proposals and log q(x)")
    st.write(dict(zip(proposals, logq.cpu().numpy().tolist())))
    st.write("#### Feature scores of proposals b(x)")
    st.write(dict(zip(proposals, feature.score(proposals))))

    # Remove ids to avoid P scoring using the wrong tokenizer
    proposals.aux_data = {}

    # Score under the EBM
    st.write("#### log P(x)")
    logP = P.log_score(proposals)
    st.write(dict(zip(proposals, logP.cpu().numpy().tolist())))

    ### QRS samples
    _ = """
    st.write("#### QRS @ 0.1 AR")
    qrs = QuasiRejectionSampler(P, q, nsamples, min_acceptance_rate=1.)
    qrs_samples = qrs.sample()
    st.write(qrs_samples)

    st.write("#### QRS @ 0.01 AR")
    qrs = QuasiRejectionSampler(P, q, nsamples, min_acceptance_rate=0.01)
    qrs_samples = qrs.sample()
    st.write(qrs_samples)
    """

if __name__ == "__main__":
    main()
