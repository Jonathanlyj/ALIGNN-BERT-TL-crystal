
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
import re

SEED = 1
props = ['formation_energy_peratom', 'ehull', 'mbj_bandgap', 'slme', 'spillage', 'magmom_outcar', 'Tc_supercon']
# props = ['formation_energy_peratom']


parser = argparse.ArgumentParser(description='run ml regressors on dataset')
# parser.add_argument('--data_path', help='path to the dataset',default=None, type=str, required=False)
parser.add_argument('--input_dir', help='input data directory', default="./data/embeddings", type=str,required=False)
# parser.add_argument('--input', help='input attributes set', default=None, type=str, required=False)
parser.add_argument('--text', help='text sources for sample', choices=['raw', 'chemnlp', 'robo', 'combo'], default='robo', type=str, required=False)
parser.add_argument('--llm', help='pre-trained llm to use', default='gpt2', type=str,required=False)
parser.add_argument('--save_data', action='store_true')
parser.add_argument('--gnn_only', action='store_true')
parser.add_argument('--gnn_file_path', type=str, required=False)
parser.add_argument('--split_dir', type=str, required=False)
parser.add_argument('--sample', action='store_true')
parser.add_argument('--skip_sentence', help='skip the ith sentence', default=None, required=False)
parser.add_argument('--mask_words', help='skip the ith word', default=None, required=False)
parser.add_argument('--prop', help="specify property from 'formation_energy_peratom', 'ehull', 'mbj_bandgap', 'slme', 'spillage', 'magmom_outcar', 'Tc_supercon'", default='all', required=False)

args =  parser.parse_args()
config = configparser.ConfigParser()

selected_samples = ["JVASP-1151"]


def prepare_dataset(args, prop):
    embeddings = []
    labels = []
    file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_*.csv"
    if args.skip_sentence is not None:
        file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_skip_{args.skip_sentence}*.csv"
    if args.mask_words is not None:
        file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_mask_{args.mask_words}*.csv"
    if args.input_dir:
        file_path = os.path.join(args.input_dir, file_path)
        print(file_path)
    embed_file = glob.glob(file_path)

    if len(embed_file)>1:
        if args.skip_sentence is None and args.mask_words is None:
            pattern_str = rf".*embeddings_{args.llm.replace('/', '_')}_{args.text}_(\d+)"
            pattern = re.compile(pattern_str)
            embed_file = [file for file in embed_file if pattern.match(file)]
        latest_file = max(embed_file, key=os.path.getctime)
        print("Latest file:", latest_file)
        embed_file = [latest_file]
    
    logging.info(f"Found embedding file: {embed_file}")
    df_embed = pd.read_csv(embed_file[0], index_col = 0)
    dat = data('dft_3d')
    ids = []


    for i in tqdm(dat, desc="Preparing data"):
        if args.sample:
            if i['jid'] not in selected_samples:
                continue
        if i[prop]!='na':
            if i['jid'] in df_embed.index:
                embeddings.append(df_embed.loc[i['jid']].values)
                labels.append(i[prop])
                ids.append(i['jid'])
    

    num_cols = len(embeddings[0])
    col_names = [i for i in range(num_cols)]
    df_data = pd.DataFrame(embeddings, columns=col_names)
    df_data[prop] = labels
    df_data["ids"] = ids
    if args.gnn_only:
        dataset_filename = f"dataset_only_prop_{prop}"
    else:
        dataset_filename = f"dataset_{args.llm.replace('/', '_')}_{args.text}_prop_{prop}"
    if args.skip_sentence is not None:
        dataset_filename = f"dataset_{args.llm.replace('/', '_')}_{args.text}_skip_{args.skip_sentence}_prop_{prop}"
    if args.mask_words is not None:
        dataset_filename = f"dataset_{args.llm.replace('/', '_')}_{args.text}_mask_{args.mask_words}_prop_{prop}"
    dataset_path = f"./data/{dataset_filename}"
    df_data['ids'] = df_data['ids'] + '.vasp'
    if args.gnn_file_path:    
        df_gnn = pd.read_csv(args.gnn_file_path)
        dataset_path = dataset_path.replace("dataset_", "dataset_alignn_")
        if args.gnn_only:
            df_gnn = pd.read_csv(args.gnn_file_path)
            df_gnn['id'] = df_gnn['id'] + '.vasp'


            df_data = df_data[[prop, "ids"]].merge(df_gnn, how='inner', left_on="ids", right_on="id", suffixes=('_lm', '_gnn'))

        else:
            df_gnn['id'] = df_gnn['id'] + '.vasp'
            df_data = df_data.merge(df_gnn, how='inner', left_on="ids", right_on="id", suffixes=('_lm', '_gnn'))
            print(df_data.head())
        df_data[prop] = df_data.pop(prop)
        df_data["ids"] = df_data.pop("ids")

    if args.split_dir:
        split_path = os.path.join(args.split_dir, f"dataset_split_{prop}.json")
        assert prop in split_path
        # for subset in ["test", "val", "train"]:
        #     sub_filename = f"{dataset_filename}_{subset}.csv"
        #     df_sub = pd.read_csv(os.path.join(args.split_dir, sub_filename))
        #     df_sub['ids'] = df_sub['ids'] + '.vasp'
        #     df_datasub = df_data[df_data["ids"].isin(df_sub["ids"])].drop(columns={"jid", "jid.1"})
        #     df_datasub.to_csv(f"{dataset_path}_{subset}.csv")
        #     logging.info(f"Saved {subset} dataset to {dataset_path}_{subset}.csv")
        with open(split_path, 'r') as json_file:
            split_dic = json.load(json_file)
        for subset in ["test", "val", "train"]:
            sub_ids = [val+'.vasp' for val in split_dic[f"id_{subset}"]]
            if not set(sub_ids).issubset(df_data['ids']):
                logging.error(f"Subset {subset} not found in GNN or LLM embedding dataset for {prop} property. Skipping...")
                return None, None
            df_datasub = df_data[df_data['ids'].isin(sub_ids)].drop(columns={"id", "full"}, errors='ignore')
            df_datasub.to_csv(f"{dataset_path}_{subset}.csv")
            print(f"{dataset_path}_{subset}: {len(df_datasub)}")
            logging.info(f"Saved {subset} dataset to {dataset_path}_{subset}.csv")
    
    else:
        logging.info(f"Constructed {df_data.shape[0]} samples for {prop} property")
        df_data.to_csv(f"{dataset_path}.csv")
        logging.info(f"Saved dataset to {dataset_path}.csv")


    return embeddings, labels

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S') 
    if args.prop != 'all':
        assert args.prop in props
        props = [args.prop]
    for prop in props:
        prepare_dataset(args, prop)

