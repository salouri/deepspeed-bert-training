#!/bin/bash

# Clone the repository
git clone https://github.com/salouri/deepspeed-bert-training.git
cd deepspeed-bert-training

# Create and activate the conda environment
conda create -n deepspeed python=3.10 -y
source activate deepspeed

# Install dependencies
pip install -r requirements.txt

# Run the script
python train_bert_ds.py --checkpoint_dir ./experiment
