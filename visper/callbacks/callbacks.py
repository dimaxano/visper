import os
import json
import shutil
from pathlib import Path
from typing import Dict, List
from collections import OrderedDict

import torch
from catalyst.dl.callbacks import CheckpointCallback
from catalyst.dl import utils
from catalyst.dl.core.callback import Callback, RunnerState
from catalyst.dl.utils.criterion import accuracy


class CheckpointCallbackV2(CheckpointCallback):
    """
    Checkpoint callback to save/restore your model/criterion/optimizer/metrics.
    """

    def __init__(
        self,
        config_path,
        save_n_best: int = 3,
        resume: str = None,
        resume_dir: str = None,
    ):
        """
        Args:
            save_n_best: number of best checkpoint to keep
            resume: path to checkpoint to load and initialize runner state
        """
        self.config_path = Path(config_path)
        self.save_n_best = save_n_best
        self.resume = resume
        self.resume_dir = resume_dir
        self.top_best_metrics = []

        self._keys_from_state = ["resume", "resume_dir"]

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
    
    def save_config(self, logdir: Path):
        self.experimental_config_path = logdir / self.config_path.name
        if self.experimental_config_path.is_file():
            return
        shutil.copy(self.config_path, str(self.experimental_config_path))

    def on_stage_start(self, state: RunnerState):
        for key in self._keys_from_state:
            value = getattr(state, key, None)
            if value is not None:
                setattr(self, key, value)

        if self.resume_dir is not None:
            self.resume = str(self.resume_dir) + "/" + str(self.resume)

        if self.resume is not None:
            self.load_checkpoint(filename=self.resume, state=state)
    

    def on_epoch_end(self, state: RunnerState):
        if state.loader_name == "train":
            self.save_config(state.logdir)


class NegativeMiningCallback(Callback):
    """
        Works only for classification task
    """
    def __init__(self, topk=(1,)):
        self.topk = topk
        self.negative_samples_filename = "negative_samples.json"

        self.samples_dict = {}
    
    def on_stage_start(self, state: RunnerState):
        self.negative_samples_path = state.logdir / self.negative_samples_filename

    
    def on_epoch_start(self, state: RunnerState):
        prev_data = self.samples_dict[str(state.epoch - 1)]["data"] if state.epoch > 0 else {}

        self.samples_dict[str(state.epoch)] = {
                "phase": state.loader_name,
                "data": prev_data
                }


    def on_batch_end(self, state: RunnerState):
        epoch_str = str(state.epoch)


        samples_dict = self.samples_dict[epoch_str]["data"]

        if state.loader_name == "valid":
            targets = state.input["targets"]
            out = state.output["logits"]
            indices = self.get_negative_idx(out, targets, self.topk)

            if len(self.topk) == 1 and self.topk[0] == 1:
                k = 1

                k_indices = indices[k-1]
                negative_samples = [Path(state.input["filenames"][idx.item()]).stem for idx in k_indices]

                # update samples dict
                for neg_sample in negative_samples:
                    if neg_sample in samples_dict:
                        samples_dict[neg_sample] += 1
                    else:
                        samples_dict[neg_sample] = 1
                
                # sort sample dict by value
                samples_dict_sorted = sorted(samples_dict.items(), key=lambda kv: kv[1], reverse=True)
                samples_dict_sorted = OrderedDict(samples_dict_sorted)
                self.samples_dict[epoch_str]["data"] = samples_dict_sorted
                
                with open(self.negative_samples_path, "w") as f:
                    json.dump(self.samples_dict, f)
            else:
                raise NotImplementedError()


    def get_negative_idx(self, outputs, targets, topk=(1, )):
        """
            Computes the accuracy@k for the specified values of k
        """
        max_k = max(topk)

        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            negative_idx = (correct[:k].view(-1).float() == 0).nonzero()
            res.append(negative_idx)
        return res


class InferenceCallback(Callback):
    def __init__(self,):
        pass
    
    def on_stage_end(self, state: RunnerState):
        infer_log_file = state.log_dir / "metrics_infer.txt"

        with open(str(infer_log_file), "w") as f:
            for k, v in state.metrics.epoch_values[state.stage].items():
                f.write(f"{k}: {v}\n")

    

    
