
import gin
import torch
from torch.utils.data import DataLoader
from catalyst.dl import SupervisedRunner

from models import LipNext
from data.dataset import LipreadingDataset

@gin.configurable
def main(
    log_dir,
    dataset_dir,
    num_epochs=50,
    num_labels=20,
    augment=True,
    batch_size=38,
    num_workers=4,
    val_batch_size_multiplier=2,
    lr=1e-3,
    weight_decay=0):
    
    
    # experiment setup
    logdir = log_dir
    num_epochs = num_epochs

    # data
    train_dataset = LipreadingDataset(
        dataset_dir,
        "train",
        num_labels=num_labels,
        augment=augment
    )

    valid_dataset = LipreadingDataset(
        dataset_dir,
        "val",
        num_labels=num_labels,
        augment=False
    )

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
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # model runner
    runner = SupervisedRunner()

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True
    )


if __name__ == "__main__":
    gin.parse_config_file("config.gin", skip_unknown=False)

    model = LipNext()

    print(model.frameLen)
    print(model.nClasses)