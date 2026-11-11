# Bi-LSTM-CRF on SROIE

## Introduction

## Installation

## Data
To download the data files, you need to go to the [official sources](https://rrc.cvc.uab.es/?ch=13&com=downloads).
  
After extracting the files, update the paths in `sroie_dataset.py`.
- `data.py` - data preparation routines  
- `sroie_dataset.py` - BIO tag and load SROIE dataset, make splits  
- `sroie_explore.py` - dataset exploration routines  

## Model
- `encoders.py` - Learnable, Word2Vec, GloVe, FastText encoders  
-- default is Learnable (not pretrained)  
-- BERT encoder is not implemented, others - implemented, but not tested yet  
- `model.py` - main BiLSTMCRF and FeatureEncoder  
-- has font size feature  
-- has typical handrafted features
- `constraints.py` - CRF constraints for BIO tags  
- `train.py` - main training loop  
-- `train_helpers.py` - shared functions, shared with `train_with_optuna.py`  
- `train_cpu.sh`, `train_gpu.sh` - cpu/gpu batch for train, if used on HPC  

## Optuna hyperparameter search
- `optuna.sh` - cpu/gpu batch for optuna  
- `train_with_optuna.py` - main training loop for optuna  

## mlflow
- `mlflow_out.py` - script to retrieve filtered results in csv  
- `mlflow_runs.csv` - exported experiment results, used in paper below  

## Citation
If you find this code useful in your research, please cite the following paper:
``` latex
@article{Gricius2026,
  title         = {Style Matters: Enhancing Key Information Extraction with Formatting Cues},
  author        = {Rolandas Gricius and Igoris Belovas},
  year          = {2026},
}
```