from pathlib import Path

import gin
import torch
from sklearn.utils.class_weight import  compute_class_weight


def get_class_weight():
    # querying params from gin-config
    dataset_path = gin.query_parameter("LipreadingDataset.directory")
    classes = gin.query_parameter("LipreadingDataset.labels")

    if not isinstance(dataset_path, Path):
        dataset_path = Path(dataset_path)

    files = dataset_path.glob("*/train/*")
    y = [f.parent.parent.name for f in files]

    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    
    weights = torch.tensor(weights, dtype=torch.float32)

    if torch.cuda.is_available():
        return weights.to("cuda")
    else:
        return weights.to("cpu")