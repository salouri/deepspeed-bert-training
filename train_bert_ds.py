import os
import time
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import deepspeed
from transformers import AutoTokenizer
from datasets import load_dataset
from functools import partial
from typing import Dict, Any
import random
import numpy as np
import fire

from utils.data_utils import masking_function, WikiTextMLMDataset
from utils.training_utils import log_dist, is_rank_0
from utils.config_utils import get_deepspeed_config
from utils.model_utils import create_model, save_model_checkpoint, load_model_checkpoint
from utils.named_pipe_utils import NamedPipeManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Named pipe paths (use environment variables)
DATA_DIR = os.path.join(os.path.dirname(__file__),"data")
DATA_PIPE_PATH = os.getenv("DATA_PIPE_PATH", os.path.join(DATA_DIR, "data_pipe"))
CHECKPOINT_PIPE_PATH = os.getenv("CHECKPOINT_PIPE_PATH",  os.path.join(DATA_DIR, "checkpoint_pipe"))
LOG_PIPE_PATH = os.getenv("LOG_PIPE_PATH",  os.path.join(DATA_DIR, "log_pipe"))

def train(
        checkpoint_dir: str = None,
        load_checkpoint_dir: str = None,
        mask_prob: float = 0.15,
        random_replace_prob: float = 0.1,
        unmask_replace_prob: float = 0.1,
        max_seq_length: int = 512,
        tokenizer: str = "roberta-base",
        num_layers: int = 12,
        num_heads: int = 12,
        ff_dim: int = 1024,
        h_dim: int = 512,
        dropout: float = 0.1,
        batch_size: int = 16,
        num_iterations: int = 2000,
        checkpoint_every: int = 500,
        log_every: int = 100,
        local_rank: int = -1,
        dtype: str = "bf16",
):
    start_time = time.time()

    # Initialize named pipe managers
    data_pipe = NamedPipeManager(DATA_PIPE_PATH)
    checkpoint_pipe = NamedPipeManager(CHECKPOINT_PIPE_PATH)
    log_pipe = NamedPipeManager(LOG_PIPE_PATH)

    # Data loading and preprocessing
    data_loading_start = time.time()
    if is_rank_0():  # Only rank 0 writes to the pipe
        wikitext_dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
        wikitext_dataset = wikitext_dataset.filter(lambda record: record["text"] != "").map(lambda record: {"text": record["text"].rstrip("\n")})
        data_pipe.write_to_pipe(wikitext_dataset)

    wikitext_dataset = data_pipe.read_from_pipe()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    masking_function_partial = partial(masking_function, tokenizer=tokenizer, mask_prob=mask_prob, random_replace_prob=random_replace_prob, unmask_replace_prob=unmask_replace_prob, max_length=max_seq_length)
    dataset = WikiTextMLMDataset(wikitext_dataset, masking_function_partial)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_loading_end = time.time()

    log_dist(f"<= Timer => Data preloading took {data_loading_end - data_loading_start:.2f} seconds", ranks=[0], level=logging.INFO)
    log_dist("Dataset Creation Done", ranks=[0], level=logging.INFO)

    log_dist("Creating Model", ranks=[0], level=logging.INFO)
    model = create_model(num_layers=num_layers, num_heads=num_heads, ff_dim=ff_dim, h_dim=h_dim, dropout=dropout)
    log_dist("Model Creation Done", ranks=[0], level=logging.INFO)

    log_dist("Creating DeepSpeed engine", ranks=[0], level=logging.INFO)
    ds_config = get_deepspeed_config(batch_size, dtype)
    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
    log_dist("DeepSpeed engine created", ranks=[0], level=logging.INFO)

    start_step = 1

    if load_checkpoint_dir is not None:
        start_checkpoint_loading = time.time()
        if is_rank_0():
            model, start_step = load_model_checkpoint(model, load_checkpoint_dir)
            checkpoint_pipe.write_to_pipe({'start_step': start_step})
        else:
            start_step = checkpoint_pipe.read_from_pipe()['start_step']
        end_checkpoint_loading = time.time()
        log_dist(f"Checkpoint loading took {end_checkpoint_loading - start_checkpoint_loading:.2f} seconds", ranks=[0], level=logging.INFO)

    log_dist(f"Total number of model parameters: {sum([p.numel() for p in model.parameters()]):,d}", ranks=[0], level=logging.INFO)
    model.train()
    losses = []
    epoch_times = []
    summary_writer = SummaryWriter(log_dir=checkpoint_dir)
    for step, batch in enumerate(data_loader, start=start_step):
        log_dist(f"Step: {step}", ranks=[0], level=logging.INFO)
        if step >= num_iterations:
            break

        epoch_start_time = time.time()

        # Move the tensors to device
        for key, value in batch.items():
            batch[key] = value.to("cpu")  # Ensure all tensors are moved to CPU
        # Forward pass
        loss = model(**batch)
        # Backward pass
        model.backward(loss)
        # Optimizer Step
        model.step()
        losses.append(loss.item())

        # Log the metrics to the named pipe
        metrics = {
            "step": step,
            "loss": np.mean(losses)
        }
        log_pipe.write_to_pipe(metrics)

        if step % log_every == 0:
            log_dist("Loss: {0:.4f}".format(np.mean(losses)),
                    ranks=[0],
                    level=logging.INFO)
            if is_rank_0():
                summary_writer.add_scalar(f"Train/loss", np.mean(losses), step)

        # Example usage of reading metrics from pipe (if any other process writes to it)
        metrics_from_pipe = log_pipe.read_from_pipe()
        if metrics_from_pipe:
            log_dist(f"Read metrics: {metrics_from_pipe}", ranks=[0], level=logging.INFO)

        if step % checkpoint_every == 0:
            start_time = time.time()
            if is_rank_0():
                model, start_step = save_model_checkpoint(model, checkpoint_dir, step)
                checkpoint_pipe.write_to_pipe({'start_step': start_step})
            else:
                start_step = checkpoint_pipe.read_from_pipe()['start_step']
            end_time = time.time()
            log_dist(f"<= Timer => Checkpoint saving took {end_time - start_time:.2f} seconds", ranks=[0], level=logging.INFO)

            log_dist("Saved model to {0}".format(checkpoint_dir),
                    ranks=[0],
                    level=logging.INFO)

        epoch_end_time = time.time()
        epoch_times.append(epoch_end_time - epoch_start_time)
        log_dist(f"<= Timer => Epoch {step} took {epoch_end_time - epoch_start_time:.2f} seconds", ranks=[0], level=logging.INFO)

    # Save the last checkpoint if not saved yet
    if step % checkpoint_every != 0:
        start_checkpoint_time = time.time()
        if is_rank_0():
            save_model_checkpoint(model, checkpoint_dir, step)
            checkpoint_pipe.write_to_pipe({'step': step})
        else:
            step = checkpoint_pipe.read_from_pipe()['step']
        end_checkpoint_time = time.time()
        log_dist(f"<= Timer => Checkpoint saving took {end_checkpoint_time - start_checkpoint_time:.2f} seconds", ranks=[0], level=logging.INFO)

        log_dist("Saved model to {0}".format(checkpoint_dir),
                ranks=[0],
                level=logging.INFO)

    training_end_time = time.time()
    average_epoch_time = sum(epoch_times) / len(epoch_times)
    log_dist(f"<= Timer => Total training time: {(training_end_time - start_time) / 60:.2f} minutes", ranks=[0], level=logging.INFO)
    return checkpoint_dir

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(0)
    random.seed(0)
    fire.Fire(train)
