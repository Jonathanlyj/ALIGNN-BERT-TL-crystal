# ALIGNN-BERT-TL

This repository contains the code for Hybrid-LLM-GNN transfer learning framework to predict materials properties using concatenated embeddings extracted from ALIGNN and BERT models. The code provides the following functions:

* Generate string representations of provided crystal samples using NLP tools: Robocystallographer and ChemNLP
* Use a pre-trained BERT/MatBERT langugae model to extract context-aware word embeddings from text descriptions of a given crystal structure
* Concatenate LLM embeddings with ALIGNN embeddings obtained from [feature-extraction-based ALIGNNTL](https://github.com/NU-CUCIS/ALIGNNTL/tree/main/FeatureExtraction)
* Post-analysis script for model performance analysis and text-based model explanation

## Installation 

The basic requirement for using the files are a Python 3.9 with the packages listed in requirements.txt. It is advisable to create a virtual environment with the correct dependencies.

## Source code for proposed approach
* [`ALIGNN_BERT_TL.ipynb`](./ALIGNN_BERT_TL.ipynb): a example notebook that sets up enviroment/installation, loads input data, extracts features and runs through the modelling pipeline with example dataset. Also accessible online at [`Google Colab`](https://colab.research.google.com/drive/1gsZoLey_M7e1e3GMOxdUrRaRh720DOlQ?usp=sharing)
* [`generator.py`](./generator.py): code to generate text descriptions for crystal samples using Robocystallographer and ChemNLP tools.
* [`preprocess.py`](./preprocess.py): code to extract contextual-aware word embeddings from text representations.
* [`feature.py`](./feature.py): code to combine LLM-based and GNN-based embeddings and construct datasets for predictor model.
* [`CrossPropertyTL`](./CrossPropertyTL): contains code for training and inferring a forward model with fully-connected layers to predict material properties. The model architecture is a straightforward MLP with fully connected layers and is not the primary focus of this research.
   * `/elemnet/dl_regressors_tf2.py`: the main model training script which trains a model based on the config file and then evaluating performance by the testset.
   * `/sample`: the config files folder that stores configuration for each run.
   * `/log`: training logs saved which saved the printouts during training & evaluating.
   * `/model`: saved model instances after each training.
   * `/prediction`: stores predictions for test set after running the train pipeline
* [`analysis`](./analysis): script code to parse predictions and create visualizations for model performance analysis and text-based model explanation.

## Other Files
* [`matbert`](./matbert): downloaded matbert llm model instance
* [`alignn_scratch`](./alignn_scratch): config files and metadata generation codes for running ALIGNN scratch model
* [`baseline`](./baseline): tree-based baseline model files
   * [`/run.py`]: Adapted from [JARVIS leaderboard](https://github.com/usnistgov/jarvis_leaderboard/blob/main/jarvis_leaderboard/contributions/matminer_xgboost/run.py). This script runs through matminer feature generation and xgboost modeling for all properties. The training and evaluation use the same dataset as in `data/split`. 
* [`chemnlp`](./chemnelp): chemnlp python library used fro generating chemnlp text
* [`data/text`](./data/text): staging csv files that contain generated text descriptions for crystal samples. Outputs from `generator.py`
* [`data/split`](./data/split): JSON files containing sample IDs used in training, validation and testing. Used in `feature.py`.
* [`data/embeddings/`](./data/embeddings/): data folder that stores generated GNN embeddings and LLM embeddings
   * `embeddings_[bert-base-uncased/matbert-base-cased]_[robo/chemnlp]_*.csv`: LLM embeddings generated from text. Outputs from `preprocess.py`. Staging text description csv file (`robo_0_75993.csv`) for robocrstallographer is available [online](https://figshare.com/articles/dataset/ALIGNN_BERT_TL_project_dataset/27115465)
   * `data0.csv`: ALIGNN embeddings generated from [ALIGNNTL: Feature Extraction](https://github.com/NU-CUCIS/ALIGNNTL.git) using the formation energy model basedon MP project dataset as source model. `data0.csv` contains all 75K dft-3d samples, which is available [online](https://figshare.com/articles/dataset/ALIGNN_BERT_TL_project_dataset/27115465) 
* `data/dataset_alignn_[bert-base-uncased/matbert-base-cased]_[robo/chemnlp]_[property]_[train/val/test].csv`: Feature dataset for forward model training to predict material properties. Outputs from `feature.py`.



## Feature Extraction and Concatenation

User can follow the steps shown below in order to create Hybrid-LLM-GNN embeddings.


1. Git clone this repo with submodules loaded
   ```
   git clone --recurse-submodules <repository-url>
   ```
2. ALIGNN-based embeddings: Download ALIGNN embeddings (staging data) for 75k dft-3d dataset [online](`https://figshare.com/s/4c190fb6fe7335bda205`) and store under `data/embeddings/`. The source model used is the formation energy model trained on MP project dataset. Alternatively, follow instructions in [ALIGNNTL: Feature Extraction](https://github.com/NU-CUCIS/ALIGNNTL.git) to extract ALIGNN-based embeddings from pre-trained ALIGNN model
   * Perform feature extraction  by running `create_features.sh` script.
   * Run the jupyter notebooks `pre-processing.ipynb` to convert the structure-wise features into a dataset. Output: `/embeddings/data0.csv`. 

3. LLM-based embeddings generation and concatenation: Once the GNN embeddings are extracted, start generating LLM feature extraction by running the 'feature_extractor' script by specifying the text source (Robocystallographer or ChemNLP) and LLM model (BERT or MatBERT):
   
   ```
   ./feature_extractor.sh [robo/chemnlp] [bert-base-uncased/matbert-base-cased]
   ```

 The script executes 3 Python programs which can also be executed seperately. It automates the generation of text files, LLM-based embeddings, and embeddings concatenation for the downstream predictor model. As the output, it prepares dataset for all material properties listed in 'feature.py' file.

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

  `python dl_regressors_tf2.py --config_file sample/example_alignn_matbert-base-cased_robo_prop_mbj_bandgap.config`

## Baseline Models for Comparison

In addition to the proposed Hybrid-GNN-LLM transfer learning method, this work involves multiple baseline models for comparison. 

* Tree-based ML regressor with Matminer features. Run the following command to reproduce the performance of xgboost tree regressor in JARVIS leaderboard  on this dataset
   ```
   python ./baseline/run.py
   ```

* ALIGNN scratch model with structure files of crystal samples. To reproduce the performance vanilla ALIGNN on this dataset, please refer to [alignn](https://github.com/usnistgov/alignn.git) codebase and follow the instructions in ths published repo. Specify the config file as in the example commands with config files under  `/alignn_scratch` folder for each property. In terms of `id_prop.csv` file and structure files `JVASP-*.vasp` required for training, please refer to `/alignn_scratch/dataset.ipynb` to generate for this dataset. Remember to create a new python env with ALIGNN repo dependencies before running ALIGNN scratch model.

* ALIGNNTL appraoch (with ALIGNN embeddings only). To reproduce the performance ALIGNNTL method on this dataset, Refer to the Abalation Study section in `ALIGNN_BERT_TL.ipynb` for detailed instructions.

## Publication

1. Youjia Li, Vishu Gupta, Muhammed Nur Talha Kilic, Kamal Choudhary, Daniel Wines, Wei-keng Liao, Alok Choudhary, and Ankit Agrawal, “Hybrid-LLM-GNN: integrating large language models and graph neural networks for enhanced materials property prediction,” Digital Discovery, 2025. [<a href="https://pubs.rsc.org/en/content/articlepdf/2025/dd/d4dd00199k">PDF</a>]
```tex
@article{li2025hybrid,
  title={Hybrid-LLM-GNN: integrating large language models and graph neural networks for enhanced materials property prediction},
  author={Li, Youjia and Gupta, Vishu and Kilic, Muhammed Nur Talha and Choudhary, Kamal and Wines, Daniel and Liao, Wei-keng and Choudhary, Alok and Agrawal, Ankit},
  journal={Digital Discovery},
  year={2025},
  publisher={Royal Society of Chemistry}
}
```

## Acknowledgements
This work was carried out with the support of the following fi-
nancial assistance award 70NANB19H005 from U.S. Department
of Commerce, National Institute of Standards and Technology as
part of the Center for Hierarchical Materials Design (CHiMaD).
Partial support is also acknowledged from NSF awards CMMI-
2053929 and OAC-2331329, DOE award DE-SC0021399, and
Northwestern Center for Nanocombinatorics. 
