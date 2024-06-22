# ALIGNN-BERT-TL

This repository contains the code for Hybrid-LLM-GNN transfer learning framework to predict materials properties using concatenated embeddings extracted from ALIGNN and BERT models. The code provides the following functions:

* Generate string representations of provided crystal samples using NLP tools: Robocystallographer and ChemNLP
* Use a pre-trained BERT/MatBERT langugae model to extract context-aware word embeddings from text descriptions of a given crystal structure
* Concatenate LLM embeddings with ALIGNN embeddings obtained from [feature-extraction-based ALIGNNTL](https://github.com/NU-CUCIS/ALIGNNTL/tree/main/FeatureExtraction)
* Post-analysis script for model performance analysis and text-based model explanation

## Installation 

The basic requirement for using the files are a Python 3.8 with the packages listed in requirements.txt. It is advisable to create a virtual environment with the correct dependencies.

## Source Files

* [`generator.py`](./generator.py): code to generate text descriptions for crystal samples using Robocystallographer and ChemNLP tools.
* [`preprocess.py`](./preprocess.py): code to extract contextual-aware word embeddings from text representations.
* [`feature.py`](./feature.py): code to combine LLM-based and GNN-based embeddings and construct datasets for predictor model.
* [`analysis`](./analysis): script code to parse predictions and create visualizations for model performance analysis and text-based model explanation
