# Mind Reading Using Multimodal Classification Machine Learning

Classifying visual object categories from EEG brain signals, image features and text features using the multimodal ThingsEEG-Text (BraVL) dataset.

## Overview

The project builds a full machine learning pipeline, from exploring the data to designing and comparing models:

- **K-Nearest Neighbours from scratch.** A KNN classifier implemented without ML libraries, benchmarked against scikit-learn's KNN (85% vs 87% accuracy).
- **Improved KNN.** Linear Discriminant Analysis applied to each modality (brain, image, text) separately, mutual-information feature selection to keep the 20 most informative features, and grid-searched hyperparameters (k = 3, uniform weights, Manhattan distance). This raised accuracy to 98%, although the gap suggested overfitting.
- **SVM ensemble.** To improve generalisation, one SVM per modality (RBF kernel for brain data, linear kernels for image and text), combined by majority vote. On a 50:50 train/test split this reached 95% accuracy and 96% precision.

The report, including figures, confusion matrices and discussion, is in [`hqcb32.pdf`](hqcb32.pdf).

## Repository structure

| File | Purpose |
|---|---|
| `KNN RAW implementation.ipynb` | From-scratch KNN on the raw features |
| `combinedRaw.ipynb` | Combines the raw brain, image and text features |
| `brain processing LDA.ipynb` | LDA and feature selection on the brain (EEG) data |
| `image processing LDA.ipynb` | LDA and feature selection on the image data |
| `text processing LDA.ipynb` | LDA and feature selection on the text data |
| `ImprovedKnn.ipynb` | KNN on the LDA-processed features, with hyperparameter tuning |
| `Ensemble.ipynb` | SVM ensemble with majority voting |
| `hqcb32.pdf` | Project report |

## Getting started

### 1. Install dependencies

Python 3.9+ with Jupyter is recommended. Install the libraries used across the notebooks:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 2. Download the dataset

The dataset is not included in this repository because of its size. From the repository's root folder, run:

```bash
mkdir -p data
curl -L https://figshare.com/ndownloader/files/36977293 -o data/ThingsEEG-Text.zip
unzip data/ThingsEEG-Text.zip -d data/
```

This creates a `data/` folder containing the extracted dataset, which is where the notebooks expect to find it.

If you're running the notebooks in Google Colab instead, put `!` in front of each command and use `%cd` rather than `cd`.

### 3. Run the notebooks

Each notebook is self-contained, so you can open any of them with `jupyter notebook` and run it from top to bottom.

To see the final model, run **`Ensemble.ipynb`**. It gives the best balance of accuracy and generalisation. The other notebooks show the earlier stages of the project and can be run on their own if you want to explore them.

## Dataset

ThingsEEG-Text is part of the BraVL dataset introduced in:

> Du, C., Fu, K., Li, J. and He, H. *Decoding Visual Neural Representations by Multimodal Learning of Brain-Visual-Linguistic Features.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.