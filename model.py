#mean_absolute_error: 54.10120434782608
#mean_absolute_error_formation_energy_peratom: 0.471877877532833
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, GPT2Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
from jarvis.db.figshare import data
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm 
import argparse
import logging
import glob
import os
import pandas as pd
from collections import defaultdict
import configparser
from .features import prepare_dataset

SEED = 1
#props = ['ehull','mbj_bandgap', 'slme', 'spillage', 'magmom_outcar','formation_energy_peratom', 'Tc_supercon']
props = ['ehull','mbj_bandgap', 'slme', 'spillage', 'magmom_outcar','formation_energy_peratom', 'Tc_supercon']


parser = argparse.ArgumentParser(description='run ml regressors on dataset')
# parser.add_argument('--data_path', help='path to the dataset',default=None, type=str, required=False)
parser.add_argument('--input_dir', help='input data directory', default="./data/embeddings", type=str,required=False)
# parser.add_argument('--input', help='input attributes set', default=None, type=str, required=False)
parser.add_argument('--text', help='text sources for sample', choices=['raw', 'chemnlp', 'robo'], default='raw', type=str, required=False)
parser.add_argument('--llm', help='pre-trained llm to use', default='gpt2', type=str,required=False)
parser.add_argument('--output_dir', help='path to the save output embedding', default=None, type=str, required=False)
parser.add_argument('--label', help='target variable', default=None, type=str,required=False)
parser.add_argument('--raw', action='store_true')
parser.add_argument('--no_save', action='store_true')
parser.add_argument('--save_data', action='store_true')
parser.add_argument('--data_only', action='store_true')
args =  parser.parse_args()
config = configparser.ConfigParser()
config.read('config.ini')




   
# Main function
def run_regressor(args):
    global props
    if args.label:
        props = [args.label] #'formation_energy_peratom'#'exfoliation_energy'
    result = defaultdict(list)
    for prop in props:
        X, y = prepare_dataset(args, prop)
        if args.data_only:
            continue
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

        # Initialize and fit a linear regression model
        logging.info(f"Started fitting to model")
        regression_model = RandomForestRegressor(n_jobs = 16, n_estimators = 1000) #LinearRegression()
        regression_model.fit(X_train, y_train)

        # Predict using the test set
        y_pred = regression_model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        logging.info(f"{prop}: mean_absolute_error: {mae}")
        # plt.plot(y_test, y_pred,'.')
        # plt.savefig('plot.png')
        # plt.close()
        logging.info(f"{prop}: Mean Squared Error: {mse}")
        result['prop'].append(prop)
        result['mae'].append(mae)
        result['mse'].append(mse)
        df_pred = pd.DataFrame({'labels': y_test, 'predictions': y_pred})
        if not args.no_save:
            df_pred.to_csv(f"./pred/rf_{args.llm.replace('/','_')}_{args.text}_{prop}.csv")
    df_rst = pd.DataFrame.from_dict(result)
    return df_rst

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S') 
    df_rst = run_regressor(args)
    filtered_str = '' if args.raw else '_filtered'
    output_csv = f"rf_{args.llm.replace('/','_')}_{args.text}_prop_{len(props)}_{filtered_str}.csv"
    if args.output_dir:
        output_csv = os.path.join(args.output_dir, output_csv)
    if not args.no_save:
        df_rst.to_csv(output_csv)

