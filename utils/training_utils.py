import time
import torch
import numpy as np
import logging
import logging
import torch
import json
import sh
import pytz
import datetime
import pathlib
from typing import Dict, Any

def log_dist(message, ranks=None, level=logging.INFO):
    if ranks is None or is_rank_0():
        logging.log(level, message)

def is_rank_0():
    return torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True

def calculate_and_log_epoch_time(start_time, end_time, step, ranks=None):
    epoch_time = end_time - start_time
    log_dist(f"<= Timer => Epoch {step} took {epoch_time / 60:.2f} minutes", ranks=ranks, level=logging.INFO)
    return epoch_time

def train_model(model, data_iterator, num_iterations, log_every, checkpoint_every, exp_dir, start_step=1):
    losses = []
    total_epoch_time = 0
    epoch_times = []

    for step, batch in enumerate(data_iterator, start=start_step):
        log_dist(f"Step: {step}", ranks=[0], level=logging.INFO)
        if step >= num_iterations:
            break

        epoch_start_time = time.time()

        # Move the tensors to device
        for key, value in batch.items():
            batch[key] = value.to("cpu")  # Ensure all tensors are moved to CPU

        loss = model(**batch) # Forward pass

        model.backward(loss) # Backward pass

        model.step() # Optimizer Step

        losses.append(loss.item())

        if step % log_every == 0:
            log_dist(f"Loss: {np.mean(losses):.4f}", ranks=[0], level=logging.INFO)
            if is_rank_0():
                summary_writer.add_scalar("Train/loss", np.mean(losses), step)

        if step % checkpoint_every == 0:
            start_time = time.time()
            model.save_checkpoint(save_dir=exp_dir, client_state={'checkpoint_step': step})
            end_time = time.time()
            log_dist(f"<= Timer => Checkpoint saving took {end_time - start_time:.2f} seconds", ranks=[0], level=logging.INFO)
            log_dist(f"Saved model to {exp_dir}", ranks=[0], level=logging.INFO)

        epoch_end_time = time.time()
        epoch_time = calculate_and_log_epoch_time(epoch_start_time, epoch_end_time, step, ranks=[0])
        total_epoch_time += epoch_time
        epoch_times.append(epoch_time)

    avg_epoch_time = total_epoch_time / len(epoch_times)
    log_dist(f"<= Timer => Average epoch time: {avg_epoch_time / 60:.2f} minutes", ranks=[0], level=logging.INFO)

    return total_epoch_time / 60  # Return total training time in minutes

def get_unique_identifier():
    return str(uuid.uuid4())