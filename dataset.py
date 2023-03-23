import torch
import config
import numpy as np
from typing import Callable
from typing_extensions import Self

class CIFARDataset(torch.utils.data.Dataset):
    def __init__(
        self: Self,
        filepath: str = "cifar-10-python.npz",
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:

        super().__init__()

        with np.load(filepath) as data:
            x: np.ndarray = np.concatenate((data["x_train"], data["x_test"]))
            y: np.ndarray = np.concatenate((data["y_train"], data["y_test"]))

            self.data: torch.Tensor = x
            # self.data = self.data.movedim(1, 3)
            self.labels: torch.Tensor = y

            # self.labels = torch.eye(10, device=config.device)[self.labels]

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self: Self) -> int:
        return len(self.labels)

    def __getitem__(self: Self, idx: int) -> tuple[torch.Tensor, int]:
        img: torch.Tensor = torch.from_numpy(self.data[idx]).to(config.device, torch.float32)
        label: int = torch.tensor(self.labels[idx], device=config.device)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label
