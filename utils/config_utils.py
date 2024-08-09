def get_deepspeed_config(batch_size: int):
    """
    Return a DeepSpeed configuration dictionary.

    Args:
        batch_size (int): The batch size for training.

    Returns:
        dict: DeepSpeed configuration dictionary.
    """
    return {
        "train_micro_batch_size_per_gpu": batch_size,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4
            }
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {
                "device": "cpu"
            }
        },
        "fp16": {
        "enabled": False
    },
    "cpu_offload": True,
    "cpu_offload_params": {
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        }
    },
    "fp32_allreduce": True,
    "disable_numa": True
    }
