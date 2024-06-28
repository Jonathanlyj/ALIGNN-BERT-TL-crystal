# ALIGNN-BERT-TL

This repository contains the code for Hybrid-LLM-GNN transfer learning framework to predict materials properties using concatenated embeddings extracted from ALIGNN and BERT models. The code provides the following functions:

* Generate string representations of provided crystal samples using NLP tools: Robocystallographer and ChemNLP
* Use a pre-trained BERT/MatBERT langugae model to extract context-aware word embeddings from text descriptions of a given crystal structure
* Concatenate LLM embeddings with ALIGNN embeddings obtained from [feature-extraction-based ALIGNNTL](https://github.com/NU-CUCIS/ALIGNNTL/tree/main/FeatureExtraction)
* Post-analysis script for model performance analysis and text-based model explanation

## Installation 

The basic requirement for using the files are a Python 3.9 with the packages listed in requirements.txt. It is advisable to create a virtual environment with the correct dependencies.

## Source Files

* [`generator.py`](./generator.py): code to generate text descriptions for crystal samples using Robocystallographer and ChemNLP tools.
* [`preprocess.py`](./preprocess.py): code to extract contextual-aware word embeddings from text representations.
* [`feature.py`](./feature.py): code to combine LLM-based and GNN-based embeddings and construct datasets for predictor model.
* [`CrossPropertyTL`](./CrossPropertyTL): contains code for training and inferring a forward DL model to predict material properties.
* [`analysis`](./analysis): script code to parse predictions and create visualizations for model performance analysis and text-based model explanation

## Feature Extraction and Concatenation

User can follow the steps shown below in order to create Hybrid-LLM-GNN embeddings.

1. Follow instructions in [ALIGNNTL: Feature Extraction](https://github.com/NU-CUCIS/ALIGNNTL.git) to extract ALIGNN-based embeddings from pre-trained model
   * Perform feature extraction  by running `create_features.sh` script
   * Run the jupyter notebooks `pre-processing.ipynb` to convert the structure-wise features into a dataset (example output GNN-embedding file in [embeddings/data0.csv](./data/embeddings/data0.csv))
2. Git clone this repo with submodules loaded
   ```
   git clone --recurse-submodules <repository-url>
    ```
3. Once the GNN embeddings are extracted, start generating LLM feature extraction by running the 'feature_extractor' script by specifying the text source (Robocystallographer or ChemNLP) and LLM model (BERT or MatBERT):
   
   ```
   ./feature_extractor.sh [robo/chemnlp] [bert-base-uncased/matbert-base-uncased]
   ```

 The script executes 3 Python programs which can also be executed seperately. It automates the generation of text files, LLM-based embeddings, and combined embeddings for the downstream predictor model. As the output, it prepares dataset for all material properties listed in 'feature.py' file.

## Predictor Model Training

Once datasets are ready, user can further train a deep learning model following the steps shown below in order to predict material properties with concatended embeddings.

1. Create config file to specify dataset path, model architecture, hyperparameters and other info in [`./CrossPropertyTL/sample/`](./CrossPropertyTL/sample/). Example config file provided [`here`](./CrossPropertyTL/sample/example_alignn_bert-base-uncased_robo_prop_mbj_bandgap.config)
2. Make sure the filepaths of generated datasets from last step are corrected entered in conf file:

```
{
   ...
   "train_data_path": "../../data/dataset_alignn_bert-base-uncased_robo_prop_mbj_bandgap_train.csv", 
   "val_data_path": "../../data/dataset_alignn_bert-base-uncased_robo_prop_mbj_bandgap_val.csv",
   "test_data_path": "../../data/dataset_alignn_bert-base-uncased_robo_prop_mbj_bandgap_test.csv",
   ...
      }
```
  
3. Pass the config file to the dl_regressors_tf2.py to start model training

  `python dl_regressors_tf2.py --config_file sample/example_alignn_bert-base-uncased_robo_prop_mbj_bandgap.config`


## Acknowledgements
