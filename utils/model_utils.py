import torch
from torch import nn
import time
from utils.training_utils import log_dist

def create_model(num_layers, num_heads, ff_dim, h_dim, dropout):
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout
        ),
        num_layers=num_layers
    )
    return model

def save_model_checkpoint(model, save_dir: str, step: int):
    start_time = time.time()
    model.save_checkpoint(save_dir=save_dir, client_state={'checkpoint_step': step})
    end_time = time.time()
    log_dist(f"<= Timer => Checkpoint saving took {end_time - start_time:.2f} seconds", ranks=[0], level=logging.INFO)

def load_model_checkpoint(model, load_dir: str):
    start_time = time.time()
    _, client_state = model.load_checkpoint(load_dir=load_dir)
    end_time = time.time()
    log_dist(f"Checkpoint loading took {end_time - start_time:.2f} seconds", ranks=[0], level=logging.INFO)
    return client_state['checkpoint_step'] + 1

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
