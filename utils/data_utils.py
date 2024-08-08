import os
import json
import time
import random
import pathlib
import datasets
import torch
import pytz
import datetime
import numpy as np
import sh
from transformers import AutoTokenizer
from functools import partial
from torch.utils.data import DataLoader, Dataset, TensorDataset

# WikiTextMLMDataset definition
class WikiTextMLMDataset(Dataset):
    def __init__(self, wikitext_dataset, masking_function):
        self.wikitext_dataset = wikitext_dataset
        self.masking_function = masking_function

    def __len__(self):
        return len(self.wikitext_dataset)

    def __getitem__(self, idx):
        record = self.wikitext_dataset[idx]
        input_ids, labels = self.masking_function(record["text"])
        return {"input_ids": input_ids, "labels": labels}

# Masking function
def masking_function(text, tokenizer, mask_prob, random_replace_prob, unmask_replace_prob, max_length):
    tokens = tokenizer.tokenize(text)
    num_to_mask = int(len(tokens) * mask_prob)
    masked_indices = random.sample(range(len(tokens)), num_to_mask)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    labels = input_ids.copy()

    for idx in masked_indices:
        if random.random() < random_replace_prob:
            input_ids[idx] = random.choice(list(tokenizer.get_vocab().values()))
        elif random.random() < unmask_replace_prob:
            pass
        else:
            input_ids[idx] = tokenizer.mask_token_id

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]

    padding_length = max_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    labels += [-100] * padding_length

    return torch.tensor(input_ids), torch.tensor(labels)

# Create experiment directory function
def create_experiment_dir(checkpoint_dir: pathlib.Path, all_arguments: dict) -> pathlib.Path:
    current_time = datetime.datetime.now(pytz.timezone("US/Pacific"))
    expname = "bert_pretrain.{0}.{1}.{2}.{3}.{4}.{5}.{6}".format(
        current_time.year,
        current_time.month,
        current_time.day,
        current_time.hour,
        current_time.minute,
        current_time.second,
        get_unique_identifier(),
    )
    exp_dir = checkpoint_dir / expname
    if not is_rank_0():
        return exp_dir
    exp_dir.mkdir(exist_ok=False)
    hparams_file = exp_dir / "hparams.json"
    with hparams_file.open("w") as handle:
        json.dump(obj=all_arguments, fp=handle, indent=2)
    try:
        gitlog = sh.git.log("-1", format="%H", _tty_out=False, _fg=False)
        with (exp_dir / "githash.log").open("w") as handle:
            handle.write(gitlog.stdout.decode("utf-8"))
    except sh.ErrorReturnCode_128:
        log_dist(
            "Seems like the code is not running from within a git repo, so hash will not be stored. However, it is strongly advised to use version control.",
            ranks=[0],
            level=logging.INFO)
    try:
        gitdiff = sh.git.diff(_fg=False, _tty_out=False)
        with (exp_dir / "gitdiff.log").open("w") as handle:
            handle.write(gitdiff.stdout.decode("utf-8"))
    except sh.ErrorReturnCode_129:
        log_dist(
            "Seems like the code is not running from within a git repo, so diff will not be stored. However, it is strongly advised to use version control.",
            ranks=[0],
            level=logging.INFO)
    tb_dir = exp_dir / "tb_dir"
    tb_dir.mkdir(exist_ok=False)
    return exp_dir

# Utility functions
def get_unique_identifier():
    return str(uuid.uuid4()).replace("-", "")

def is_rank_0():
    return torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True

def log_dist(message, ranks=None, level=logging.INFO):
    if ranks is None or is_rank_0():
        print(message)

