import sys
from collections import OrderedDict
from pathlib import Path

import gin
import torch
from torch.utils.data import DataLoader
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback

from .data import LipreadingDataset
from .utils import add_external_configurables
from .models import LipNext
from .callbacks import NegativeMiningCallback, InferenceCallback, CheckpointCallbackV2


# @TODO: add loading checkpoint
def load_experiment_config(experiment_name: str, log_dir: Path):
    add_external_configurables()

    config_path = log_dir / experiment_name / "config.gin"
    gin.parse_config_file(str(config_path), skip_unknown=True) 

    return config_path


def infer(
    config_path,
    log_dir
    ):
    """
        Inference:
            1. loaders
            2. model
    """

    # quering params from experiment config
    batch_size = 116


    test_dataset = LipreadingDataset(
        "test")

    loaders = {
        "infer": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,)
    }

    model = LipNext()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = SupervisedRunner(device=device)

    runner.infer(
        model=model,
        loaders=loaders,
        callbacks=[
            AccuracyCallback(accuracy_args=[1, 3]),
            InferenceCallback(),
            CheckpointCallbackV2(
                config_path=config_path,
                resume=("/home/dmitry.klimenkov/Documents/projects/visper_pytorch/logdir"
                    "/Mobi-VSR-5W-mixed_aligned_patience5_sometests/checkpoints/train.0.35.8553.pth"))
            # NegativeMiningCallback()
        ],
        state_kwargs={
            "log_dir": log_dir
        },
        check=True
    )

    

if __name__ == "__main__":
    log_dir = Path("./logdir")
    experiment_name = sys.argv[1]
    
    config_path = load_experiment_config(experiment_name, log_dir)
    infer(
        config_path=str(config_path),
        log_dir=log_dir)