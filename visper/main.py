import gin
import torch
from torch.utils.data import DataLoader
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, LRFinder
from catalyst.dl.callbacks import InferCallback, VerboseLogger

from .models import LipNext
from .data.dataset import LipreadingDataset
from .utils import add_external_configurables
from .callbacks import CheckpointCallbackV2, NegativeMiningCallback


@gin.configurable(blacklist=["config_path"])
def main(
    config_path,
    log_dir=None,
    experiment_name=None,
    dataset_dir=None,
    num_epochs=50,
    num_labels=20,
    batch_size=38,
    num_workers=4,
    val_batch_size_multiplier=2,
    lr=1e-3,
    scheduler=None,
    optimizer=None,
    weight_decay=0,
    class_weight=None,
    check=False,
    verbose=True,
    cudnn_benchmark=True):

    if torch.cuda.is_available() and cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    
    # experiment setup
    logdir = log_dir + experiment_name
    num_epochs = num_epochs

    # data
    train_dataset = LipreadingDataset(
        phase = "train")

    valid_dataset = LipreadingDataset(
        phase = "val")

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True),

        "valid": DataLoader(
            valid_dataset,
            batch_size=val_batch_size_multiplier*batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False)
    }

    # model, criterion, optimizer
    model = LipNext()

    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler:
        scheduler = scheduler(optimizer)
    else:
        scheduler = scheduler
    
    # model runner
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    runner = SupervisedRunner(device=device)

    # callbacks
    acc_callback = AccuracyCallback(accuracy_args=[1, 3])
    ckpt_callback = CheckpointCallbackV2(config_path=config_path)
    neg_mining_callback = NegativeMiningCallback()
    callbacks = [acc_callback, ckpt_callback, neg_mining_callback]


    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks,
        loaders=loaders,
        logdir=logdir,
        main_metric="accuracy01",
        minimize_metric=False,
        num_epochs=num_epochs,
        verbose=verbose,
        check=check
    )


if __name__ == "__main__":
    CONFIG_PATH = "./main_config.gin"

    add_external_configurables()
    gin.parse_config_file(CONFIG_PATH, skip_unknown=False)
    

    main(CONFIG_PATH)