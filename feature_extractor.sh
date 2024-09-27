#!/bin/bash
# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <text_value> <llm_value> <prop>"
  exit 1
fi

TEXT_VALUE=$1
LLM_VALUE=$2
PROP_VALUE=$3

# Run the python scripts with the provided arguments
python generater.py --text "$TEXT_VALUE" # Use --end k to select an small subset (the first k samples)
python preprocess.py --llm "$LLM_VALUE" --text "$TEXT_VALUE" --cache_csv "./data/text/${TEXT_VALUE}_skip_none.csv"
python features.py --gnn_file_path "./data/embeddings/data0.csv" --split_dir "./data/split/" --llm "$LLM_VALUE" --text "$TEXT_VALUE" --prop "$PROP_VALUE"