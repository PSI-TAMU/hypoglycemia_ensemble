# Hypoglycemia prediction with ECG beat ensembles

<b> Authors: <a href='https://morris88826.github.io/#/'>Mu-Ruei Tseng</a>, Kathan Vyas, Anurag Das, Waris Quamer, Darpit Dave, Madhav Erranguntla, Ricardo Gutierrez-Osuna</b>

[[paper]()] [[pre-trained weights]()]

This paper has been accepted for publication in the *Journal of Diabetes Science and Technology*.

## Overview
TODO

## Table of Contents
- [Quick Start](#quick-start)
  - [Installation](#1-installation)
  - [Download Data](#2-download-data)
  - [Training (optional)](#3-training-optional)
  - [Evaluation](#4-evaluation)
- [Results](#results)
- [Citation](#citation)

## Quick Start
### 1. Installation
Clone the repository and setup the environment
```
git clone https://github.com/PSI-TAMU/hypoglycemia_ensemble.git
cd hypoglycemia_ensemble
conda create -n ecg python=3.11.5
pip install -r requirements.txt
```

### 2. Download Data
The dataset used for the experiment is from [PhysioCGM](https://github.com/PSI-TAMU/PhysioCGM). To access the full dataset, please visit the website and follow the download instructions. Here, we provide the processed data used for training and evaluation. Please download the data from [here]().

### 3. Training (optional) 
Coming Soon.

### 4. Evaluation
We provided two jupyter notebooks for demonstrating the evaluation result:
* Beat-Level Detection: [here](./notebooks/test.ipynb)
* CGM-Level Detection (ensemble): [here](./notebooks/test_ensemble.ipynb)

<b>Note that we have provided the pre-trained weights for each model (you can download them from [here]())</b>. Put the checkpoints folder under the root directory. 

## Results
TODO

## Citation
TODO