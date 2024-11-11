#mean_absolute_error: 54.10120434782608
#mean_absolute_error_formation_energy_peratom: 0.471877877532833
import json
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig, BertTokenizerFast, BertForMaskedLM
from transformers import AutoTokenizer, GPT2Model, BertModel, OPTModel
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
import pandas as pd
import os
# import chemnlp
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from chemnlp.chemnlp.utils.describe import atoms_describer
from robocrys import StructureCondenser, StructureDescriber
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.analysis.diffraction.xrd import XRD
from jarvis.core.specie import Specie
from collections import defaultdict
import nltk
import re
nltk.download('punkt')


print(transformers.__version__)
SEED = 1

parser = argparse.ArgumentParser(description='get embeddings on dataset')
# parser.add_argument('--data_path', help='path to the dataset',default=None, type=str, required=False)
parser.add_argument('--label', help='output variable', default=None, type=str,required=False)
# parser.add_argument('--input', help='input attributes set', default=None, type=str, required=False)
parser.add_argument('--text', help='text sources for sample', choices=['raw', 'chemnlp', 'robo', 'combo'],default='raw', type=str, required=False)
parser.add_argument('--llm', help='pre-trained llm to use', default='gpt2', type=str,required=False)
parser.add_argument('--output_dir', help='path to the save output embedding', default="./data/embeddings//", type=str, required=False)
parser.add_argument('--cache_csv', help='path that stores text', default=None, type=str, required=False)
parser.add_argument('--skip_sentence', help='text topic to skip', default=None, required=False)
parser.add_argument('--mask_words', action='store_true', help='use cpu only', required=False)
parser.add_argument('--sample', action='store_true', help='save samples from lists only', required=False)
parser.add_argument('--cpu', action='store_true', help='use cpu only', required=False)
args,_ = parser.parse_known_args()

selected_samples = ["JVASP-1151"]
# words_index = [2,10]
# words_index = [31]
# words_index = [37]
# words_index = [44]
# words_index = [60]


def extract_chemnlp(text):
    # Split text into sections based on the starting phrases
    sections = re.split(r'(The chemical information include|The structure information include|The bond distances are)', text)

    # Remove empty strings and clean up whitespace
    sections = [section.strip() for section in sections if section.strip()]
    combined_data = []
    for i in range(0, len(sections), 2):
        combined_data.append(sections[i] + sections[i + 1])

    return combined_data

def mask_words(sentence: str, indexes: list) -> str:
    # Regular expression to match words and punctuation
    words = sentence.split()
    
    # Replace words at specified indexes with [MASK]
    for index in indexes:
        if 0 <= index < len(words):
            words[index] = '[MASK]'
    
    # Join the words back into a single string, ensuring spaces are added appropriately
    return ' '.join(words)


def remove_sentence_by_index(input_string, criteria, input_type=None):
    filtered_sentences = []
    if input_type=="robo":
        sentences = nltk.sent_tokenize(input_string)
        
        if type(criteria) == int: 
            filtered_sentences = [sent.strip() for i, sent in enumerate(sentences) if i != criteria]
            return ' '.join(filtered_sentences)
        else:
            assert criteria in ['summary', 'site', 'bond', 'length', 'angle']

            for i, sent in enumerate(sentences):
                if criteria == 'summary' and (i == 0 or ('structure' in sent.lower() and i == 1)):
                    # Skip the first sentence (summary sentence)
                    continue
                if criteria == 'site' and 'sites' in sent.lower() and 'there are' in sent.lower():
                    # Skip sentences containing 'sites' (site info)
                    continue
                if criteria == 'bond' and 'bonded' in sent.lower():
                    # Skip sentences containing 'bonded' (bond description)
                    continue
                if criteria == 'length' and ('bond length' in sent.lower() or 'bond distance' in sent.lower()):
                    # Skip sentences containing 'bond lengths' (bond lengths)
                    continue
                if criteria == 'angle' and ('tilt' in sent.lower() or 'angle' in sent.lower()):
                    continue
                filtered_sentences.append(sent.strip())

    elif input_type=="chemnlp":
        assert criteria in ['structure', 'chemical', 'bond']
        sentences = extract_chemnlp(input_string)
        for i, sent in enumerate(sentences):
            if criteria == 'structure' and sent.startswith('The structure information include:'):
                # Skip the first sentence (summary sentence)
                continue
            if criteria == 'chemical' and sent.startswith('The chemical information include:'):
                # Skip sentences containing 'lattice' (lattice info)
                continue
            if criteria == 'bond' and sent.startswith('The bond distances are as follows:'):
                # Skip sentences containing 'atoms' (atoms info)
                continue
            filtered_sentences.append(sent.strip())
    return ' '.join(filtered_sentences)

        


def get_crystal_string_t(atoms):
    lengths = atoms.lattice.abc  # structure.lattice.parameters[:3]
    angles = atoms.lattice.angles
    atom_ids = atoms.elements
    frac_coords = atoms.frac_coords

    crystal_str = (
        " ".join(["{0:.2f}".format(x) for x in lengths])
        + "#\n"
        + " ".join([str(int(x)) for x in angles])
        + "@\n"
        + "\n".join(
            [
                str(t) + " " + " ".join(["{0:.3f}".format(x) for x in c]) + "&"
                for t, c in zip(atom_ids, frac_coords)
            ]
        )
    )

    crystal_str = atoms_describer(atoms) + "\n*\n" + crystal_str
    return crystal_str

def atoms_describer(
    atoms=[], xrd_peaks=5, xrd_round=1, cutoff=4, take_n_bomds=2,include_spg=True
):
    """Describe an atomic structure."""
    if include_spg:
       spg = Spacegroup3D(atoms)
    theta, d_hkls, intens = XRD().simulate(atoms=(atoms))
    #     x = atoms.atomwise_angle_and_radial_distribution()
    #     bond_distances = {}
    #     for i, j in x[-1]["different_bond"].items():
    #         bond_distances[i.replace("_", "-")] = ", ".join(
    #             map(str, (sorted(list(set([round(jj, 2) for jj in j])))))
    #         )
    dists = defaultdict(list)
    elements = atoms.elements
    for i in atoms.get_all_neighbors(r=cutoff):
        for j in i:
            key = "-".join(sorted([elements[j[0]], elements[j[1]]]))
            dists[key].append(j[2])
    bond_distances = {}
    for i, j in dists.items():
        dist = sorted(set([round(k, 2) for k in j]))
        if len(dist) >= take_n_bomds:
            dist = dist[0:take_n_bomds]
        bond_distances[i] = ", ".join(map(str, dist))
    fracs = {}
    for i, j in (atoms.composition.atomic_fraction).items():
        fracs[i] = round(j, 3)
    info = {}
    chem_info = {
        "atomic_formula": atoms.composition.reduced_formula,
        "prototype": atoms.composition.prototype,
        "molecular_weight": round(atoms.composition.weight / 2, 2),
        "atomic_fraction": (fracs),
        "atomic_X": ", ".join(
            map(str, [Specie(s).X for s in atoms.uniq_species])
        ),
        "atomic_Z": ", ".join(
            map(str, [Specie(s).Z for s in atoms.uniq_species])
        ),
    }
    struct_info = {
        "lattice_parameters": ", ".join(
            map(str, [round(j, 2) for j in atoms.lattice.abc])
        ),
        "lattice_angles": ", ".join(
            map(str, [round(j, 2) for j in atoms.lattice.angles])
        ),
        #"spg_number": spg.space_group_number,
        #"spg_symbol": spg.space_group_symbol,
        "top_k_xrd_peaks": ", ".join(
            map(
                str,
                sorted(list(set([round(i, xrd_round) for i in theta])))[
                    0:xrd_peaks
                ],
            )
        ),
        "density": round(atoms.density, 3),
        #"crystal_system": spg.crystal_system,
        #"point_group": spg.point_group_symbol,
        #"wyckoff": ", ".join(list(set(spg._dataset["wyckoffs"]))),
        "bond_distances": bond_distances,
        #"natoms_primitive": spg.primitive_atoms.num_atoms,
        #"natoms_conventional": spg.conventional_standard_structure.num_atoms,
    }
    if include_spg:
        struct_info["spg_number"]=spg.space_group_number
        struct_info["spg_symbol"]=spg.space_group_symbol
        struct_info["crystal_system"]=spg.crystal_system
        struct_info["point_group"]=spg.point_group_symbol
        struct_info["wyckoff"]=", ".join(list(set(spg._dataset["wyckoffs"])))
        struct_info["natoms_primitive"]=spg.primitive_atoms.num_atoms
        struct_info["natoms_conventional"]=spg.conventional_standard_structure.num_atoms
    info["chemical_info"] = chem_info
    info["structure_info"] = struct_info
    line = "The number of atoms are: "+str(atoms.num_atoms) +". " #, The elements are: "+",".join(atoms.elements)+". "
    for i, j in info.items():
        if not isinstance(j, dict):
            line += "The " + i + " is " + j + ". "
        else:
            #print("i",i)
            #print("j",j)
            for ii, jj in j.items():
                tmp=''
                if isinstance(jj,dict):
                   for iii,jjj in jj.items():
                        tmp+=iii+": "+str(jjj)+" "
                else:
                   tmp=jj
                line += "The " + ii + " is " + str(tmp) + ". "
    return line

def get_robo(structure=None):
    describer = StructureDescriber()
    condenser = StructureCondenser()
    condensed_structure = condenser.condense_structure(structure)
    description = describer.describe(condensed_structure)
    return description

def preprocess_data(args):
    dat = data('dft_3d')
    dd = []
    if args.label:
        prop = args.label
        for i in dat:
            if i[prop]!='na': 
                dd.append(i)
        logging.info(f"Filter based on {prop} property: {len(dd)} samples left")
    else:
        dd = dat
        logging.info(f"Use full dataset: {len(dd)} samples")
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = args.llm
    if llm == "matbert-base-cased":
        tokenizer = BertTokenizerFast.from_pretrained(os.path.join("./matbert", llm), do_lower_case=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(llm)
    #model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if "gpt2" in llm.lower():
        model = GPT2Model.from_pretrained(llm)
    elif "bert" in llm.lower():
        try:
            model = BertModel.from_pretrained(llm)
        except:
            model = BertModel.from_pretrained(os.path.join("./matbert", llm))
        
    elif "opt" in llm.lower():
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-6.7b", 
            device_map='auto',
            quantization_config=quantization_config
        )

    model.to(device)
    embeddings = []
    samples=[]
    # print(model)
    max_token_length = model.config.max_position_embeddings
    logging.info(f"Max token length: {max_token_length}")
    if args.cache_csv:
        assert args.text in args.cache_csv
        df_text = pd.read_csv(args.cache_csv, index_col = 'jid')

    for entry in tqdm(dat, desc="Processing data"):
        if args.sample:
            if entry['jid'] not in selected_samples:
                continue
        if args.cache_csv:
            if entry['jid'] in df_text.index:
                text = df_text.at[entry['jid'],'text']
            else:
                continue
        else:
            if args.text == 'raw':
                text = Poscar(Atoms.from_dict(entry['atoms'])).to_string()
            elif args.text == 'chemnlp':
                text = json.dumps(atoms_describer(atoms=Atoms.from_dict(entry['atoms'])))
            elif args.text == "robo":
                try:
                    text = get_robo(Atoms.from_dict(entry['atoms']).pymatgen_converter())
                except Exception as exp: 
                    pass
            elif args.text == "combo":
                text = get_crystal_string_t(Atoms.from_dict(entry['atoms']))

        if args.skip_sentence is not None:
            text = remove_sentence_by_index(text, args.skip_sentence, args.text)
        elif args.mask_words:
            text = mask_words(text, words_index)
            print(text)
        inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
        if len(inputs['input_ids'][0]) <= max_token_length:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output = model(**inputs)
            if device.type == 'cuda':
                emb = output.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            else:
                emb = output.last_hidden_state.mean(dim=1).numpy().flatten()
            embeddings.append(emb)
            samples.append(entry['jid'])

    embeddings = np.vstack(embeddings)
    #labels = np.array([entry['exfoliation_energy'] for entry in dat])

    return embeddings, samples

def save_data(embeddings, samples):
    n = len(embeddings)
    assert n == len(samples)
    df = pd.DataFrame(embeddings, index=samples)

    n = len(df)
    file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_{n}.csv"
    if args.skip_sentence is not None:
        file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_skip_{args.skip_sentence}_{n}.csv"
    if args.mask_words:
        file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_mask_{words_index[0]}_{n}.csv"
    if args.output_dir:
        file_path = os.path.join(args.output_dir, file_path) 
    df.to_csv(file_path)
    logging.info(f"Saved to {file_path}")




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    embeddings, samples = preprocess_data(args)
    logging.info(f"Finished generate embeddings")
    save_data(embeddings, samples)
    

