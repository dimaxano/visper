import os
from typing import Dict


from catalyst.dl.callbacks import CheckpointCallback
from catalyst.dl import utils
from catalyst.dl.core.callback import Callback


class CheckpointCallbackV2(CheckpointCallback):
    """
    Checkpoint callback to save/restore your model/criterion/optimizer/metrics.
    """

    def save_checkpoint(
        self,
        logdir: str,
        checkpoint: Dict,
        is_best: bool,
        save_n_best: int = 5,
        main_metric: str = "loss",
        minimize_metric: bool = True
    ):
        main_metric_val = checkpoint["valid_metrics"][main_metric]
        suffix = f"{checkpoint['stage']}.{checkpoint['epoch']}.{main_metric_val:.4f}"
        filepath = utils.save_checkpoint(
            logdir=f"{logdir}/checkpoints/",
            checkpoint=checkpoint,
            suffix=suffix,
            is_best=is_best,
            is_last=True
        )

        checkpoint_metric = checkpoint["valid_metrics"][main_metric]
        self.top_best_metrics.append((filepath, checkpoint_metric))
        self.top_best_metrics = sorted(
            self.top_best_metrics,
            key=lambda x: x[1],
            reverse=not minimize_metric
        )
        if len(self.top_best_metrics) > save_n_best:
            last_item = self.top_best_metrics.pop(-1)
            last_filepath = last_item[0]
            os.remove(last_filepath)