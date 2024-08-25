#!/bin/bash
# conda
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# basic
mamba create -n LLM-based python=3.11 -y; mamba activate LLM-based
pip install -r requirements.txt

# chrisbase
rm -rf chrisbase*; git clone git@github.com:chrisjihee/chrisbase.git
pip install --editable chrisbase*

# list
pip list | grep -E "langchain|transformers|torch|faiss|groq|replicate|together|chris"
