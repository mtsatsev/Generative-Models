from typing import Literal, Dict, Optional
import torch
from torch import optim
from torch.nn import Module
from torch.utils.data import DataLoader


class ImageTrainer:
    def __init__(
        self,
        model: Module,
        train_data: DataLoader,
        eval_data: DataLoader,
        test_data: DataLoader,
        optimizer: optim,
        device: Literal["gpu", "cpu"],
        metrics: Optional[Dict[str, callable]] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        pass
