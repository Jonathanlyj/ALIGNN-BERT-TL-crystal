# ALIGNN-BERT-TL

This repository contains the code for Hybrid-LLM-GNN transfer learning framework to predict materials properties using concatenated embeddings extracted from ALIGNN and BERT models. The code provides the following functions:

* Generate string representations of provided crystal samples using NLP tools: Robocystallographer and ChemNLP
* Use a pre-trained BERT/MatBERT langugae model to extract context-aware word embeddings from text descriptions of a given crystal structure
* Concatenate LLM embeddings with ALIGNN embeddings obtained from [feature-extraction-based ALIGNNTL](https://github.com/NU-CUCIS/ALIGNNTL/tree/main/FeatureExtraction)
* Post-analysis script for model performance analysis and text-based model explanation

## Installation 

The basic requirement for using the files are a Python 3.9 with the packages listed in requirements.txt. It is advisable to create a virtual environment with the correct dependencies.

## Source code

* [`generator.py`](./generator.py): code to generate text descriptions for crystal samples using Robocystallographer and ChemNLP tools.
* [`preprocess.py`](./preprocess.py): code to extract contextual-aware word embeddings from text representations.
* [`feature.py`](./feature.py): code to combine LLM-based and GNN-based embeddings and construct datasets for predictor model.
* [`CrossPropertyTL`](./CrossPropertyTL): contains code for training and inferring a forward DL model to predict material properties.
* [`analysis`](./analysis): script code to parse predictions and create visualizations for model performance analysis and text-based model explanation

## Other Files
* [`matbert`](./matbert): downloaded matbert llm model instance
* [`chemnlp`](./chemnelp): chemnlp python library used fro generating chemnlp text
* [`data/text`](./data/text): staging csv files that contain generated text descriptions for crystal samples. Outputs from `generator.py`
* [`data/split`](./data/split): JSON files containing sample IDs used in training, validation and testing. Used in `feature.py`.
* [`data/embeddings/`](./data/embeddings/): data folder that stores generated GNN embeddings and LLM embeddings
   * `embeddings_[bert-base-uncased/matbert-base-cased]_[robo/chemnlp]_*.csv`: LLM embeddings generated from text. Outputs from `preprocess.py`. Staging text description csv for robo is available [online](https://figshare.com/s/9bc5ddc20c10362fa0e5)
   * `data0.csv`: ALIGNN embeddings generated from [ALIGNNTL: Feature Extraction](https://github.com/NU-CUCIS/ALIGNNTL.git). `data0.csv` contains all 75K dft-3d samples, which is available [online](https://figshare.com/s/4c190fb6fe7335bda205) 
* [`data/dataset_alignn_[bert-base-uncased/matbert-base-cased]_[robo/chemnlp]_[property]_[train/val/test].csv`]: Feature dataset for forward model training to predict material properties. Outputs from `feature.py`.



## Feature Extraction and Concatenation

User can follow the steps shown below in order to create Hybrid-LLM-GNN embeddings.


1. Git clone this repo with submodules loaded
   ```
   git clone --recurse-submodules <repository-url>
   ```
2. Download ALIGNN embeddings for 75k dft-3d dataset [online](`https://figshare.com/s/4c190fb6fe7335bda205`) and store under `data/embeddings/`. Or follow instructions in [ALIGNNTL: Feature Extraction](https://github.com/NU-CUCIS/ALIGNNTL.git) to extract ALIGNN-based embeddings from pre-trained ALIGNN model
   * Perform feature extraction  by running `create_features.sh` script
   * Run the jupyter notebooks `pre-processing.ipynb` to convert the structure-wise features into a dataset (example output GNN-embedding file in [embeddings/data0_spillage.csv](./data/embeddings/data0_spillage.csv, which contains all samples needed for property Spillage))

3. Once the GNN embeddings are extracted, start generating LLM feature extraction by running the 'feature_extractor' script by specifying the text source (Robocystallographer or ChemNLP) and LLM model (BERT or MatBERT):
   
   ```
   ./feature_extractor.sh [robo/chemnlp] [bert-base-uncased/matbert-base-cased]
   ```

 The script executes 3 Python programs which can also be executed seperately. It automates the generation of text files, LLM-based embeddings, and combined embeddings for the downstream predictor model. As the output, it prepares dataset for all material properties listed in 'feature.py' file.

## Predictor Model Training

Once datasets are ready, user can further train a deep learning model following the steps shown below in order to predict material properties with concatended embeddings.

1. Create config file to specify dataset path, model architecture, hyperparameters and other info in [`./CrossPropertyTL/sample/`](./CrossPropertyTL/sample/). Example config file provided [`here`](https://github.com/Jonathanlyj/CrossPropertyTL/blob/7e39ae4f8bde8031bd99e7b5bd81ee9c6ab9f3b4/elemnet/sample/example_alignn_bert-base-uncased_robo_prop_mbj_bandgap.config)
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
This work was carried out with the support of the following fi-
nancial assistance award 70NANB19H005 from U.S. Department
of Commerce, National Institute of Standards and Technology as
part of the Center for Hierarchical Materials Design (CHiMaD).
Partial support is also acknowledged from NSF awards CMMI-
2053929 and OAC-2331329, DOE award DE-SC0021399, and
Northwestern Center for Nanocombinatorics. 
