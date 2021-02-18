# Code for "An approximate sampler for energy-based models with divergence diagnostics"

This is the source code to replicate the experiments appearing in the paper
[_"An approximate sampler for energy-based models with divergence diagnostics."_](https://openreview.net/pdf?id=VW4IrC0n0M)
Bryan Eikema, Germán Kruszewski, Christopher R Dance, Hady Elsahar, Marc Dymetman. In TMLR November 2022.



## Requirements

To install the required libraries to run this code run:
```
pip install -r requirements.txt
```
Most visualizations are written as [streamlit](https://streamlit.io/) scripts. 
To run them, please refer to https://docs.streamlit.io/library/get-started/main-concepts.

## Experiments

### Two poissons

These experiments can be run stand-alone with:
- `streamlit run qrs_tvd_toy.py`

For the comparison with MCMC techniques on the two poissons experiments, run:
- `python fish.py`

### Generation with Distributional Control

#### Preliminary steps

Many of these experiments require having access to a proposal distribution
that has been fine-tuned to approximate the target EBM using the DPG method.

To generate the DPG fine-tuned proposal distributions on the EBMs proposed by
Khalifa et al. (2021), see https://github.com/naver/gdc. Place the trained
checkpoints in the following locations where `<proposals-dir>` corresponds to
the directory where these files are stored.

- `<proposals-dir>/amazing.pt`
- `<proposals-dir>/wikileaks.pt`
- `<proposals-dir>/female.pt`
- `<proposals-dir>/female_science.pt`
- `<proposals-dir>/female_sports.pt`

Then, replace the `<proposals-dir>` placeholder in all configuration files 
with the corresponding directory, e.g., using:

`for fn in config/*/*.yml; do sed -i "s/<proposals-dir>/\/home\/my-user\/my-proposals-dir/g" $fn; done`


The scripts will use these and other models to generate samples and score them
according to the models and the EBMs.
This data will be stored in an `<output-dir>` that must be set, e.g., as follows:

`for fn in src/command-lines/*.sh; do sed -i "s/<output-dir>/\/home\/my-user\/my-output-dir/g" $fn; done`
`for fn in *.py; do sed -i "s/<output-dir>/\/home\/my-user\/my-output-dir/g" $fn; done`


#### Pointwise and distributional constraints

1. Generate samples running all commands in `command-lines/gdc-sampling.sh`
2. Score samples running all commands in `command-lines/gdc-scoring.sh`
3. Proposal distributions for a pointwise constraint: `streamlit run compare_proposals.py`
4. Debiasing and other EBM experiments: `streamlit run show-plots.py`
5. Variance estimates: `streamlit run variance-table.py`

#### MCMC comparison

1. Generate samples running all commands in `command-lines/mcmc_comparison-sampling.sh`
2. Score samples running all commands in `command-lines/mcmc_comparison-scoring.sh`
3. Evaluate samples running all commands in `command-lines/mcmc_comparison-proxy_metrics.sh`
4. Collect results with `streamlit run collect_results.py`
5. Observe examples with `streamlit run collect_examples.py`

### Paraphrasing: 
1. Sample and score paraphrases from the proposal: `sample_paraphrases.py`
2. TVD plots: `streamlit run paraphrasing.py`
3. Generating samples at $10^{-5}$ acceptance rate using QRS: `qrs_paraphrases.py`

## Bibtex

For citations, please use:
```
@article{
eikema2022an,
title={An approximate sampler for energy-based models with divergence diagnostics},
author={Bryan Eikema and Germ{\'a}n Kruszewski and Christopher R Dance and Hady Elsahar and Marc Dymetman},
journal={Transactions on Machine Learning Research},
year={2022},
url={https://openreview.net/forum?id=VW4IrC0n0M},
note={}
}
```
## License

```
qrs
Copyright (C) 2022-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

A summary of the CC BY-NC-SA 4.0 license is located here:
	https://creativecommons.org/licenses/by-nc-sa/4.0/
```

