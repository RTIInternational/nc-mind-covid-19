# COVID-19

## ℹ️ Project Summary

We have created a geospatially explicit agent-based model (ABM) to simulate hospital capacity for North Carolina (NC) hospitals during the coronavirus disease 2019 (COVID-19) pandemic. This ABM is used to predict capacity strain on both ICU and non-ICU beds for both COVID-19 and non-COVID-19 patients. The public version of this model in this repository is not fully calibrated to actual data because of data restrictions. Specifically, access to counts of COVID-19 and non-COVID-19 patients by hospital is restricted. **Without 
access to the actual data, this repository is not intended to be used to reproduce actual estimates but instead intended
to illustrate the theoretical components of the model.**

## ⚠️ Model Notes

- This code is released with no support. 
- This repository is copied from an internal repository and the commit history is removed, so the development history is not included in the public repository.

## 👟 Production Run with Parallel Processing

Due to the time it takes to run the model, we recommend running production runs using parallel processing on multiple CPUs. Follow the instructions in [scripts/README.md](scripts/README.md) before launching a production run.

A production run requires a local installation of python to create the necessary files. Using your preferred virtual environment mechanism (virtualenv, conda, etc), create the environment first by install the requirements:

```bash
pip install -r docker/requirements.txt
```

The command used to launch the production run from root while in the virtualenv is:

```
bash ./scripts/run_all.sh
```

## 📕 Documentation

The Overview, Design Concepts, Details (ODD) Protocol document is being publihed on [arXiv](https://arxiv.org/) and will be available shortly.  

## ▶️ Next Steps

This page serves as an introduction to the project. Please see this [`README.md`](model_runs/README.md) for model details. 

