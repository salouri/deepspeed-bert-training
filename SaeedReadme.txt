DeepSpeed BERT Training

This repository contains the setup and scripts to train a BERT model using DeepSpeed.

Setup

Step 1: Clone the Repository

git clone https://github.com/yourusername/deepspeed-bert-training.git
cd deepspeed-bert-training

Step 2: Create and Activate Conda Environment

conda create -n deepspeed python=3.10
conda activate deepspeed

Step 3: Install Dependencies

pip install -r requirements.txt

Step 4: Run the Script

python train_bert_ds.py --checkpoint_dir ./experiment

Dependencies

- Python 3.10
- DeepSpeed
- mpi4py
- Other dependencies listed in requirements.txt
