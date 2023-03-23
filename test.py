import torch
from network import NeuralNetwork
from dataset import CIFARDataset
from matplotlib import pyplot as plt
import numpy as np


def main() -> None:
    model: NeuralNetwork = torch.load("trained_model.pt")

    LABELS: tuple[str] = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    dataset: CIFARDataset = CIFARDataset()

    _, subplt = plt.subplots(5, 5)

    inp = torch.stack([dataset[i][0] for i in range(25)])
    labels = torch.stack([dataset[i][1] for i in range(25)])

    out = torch.argmax(model(inp), dim=1)

    for i in range(5):
        for j in range(5):
            actual = LABELS[labels[5 * i + j]]
            pred = LABELS[out[5 * i + j]]
            img: np.ndarray = (
                torch.permute(inp[5 * i + j], (1, 2, 0)).cpu().numpy().astype("int")
            )
            print(actual, pred)
            subplt[i][j].axis("off")
            subplt[i][j].imshow(img)
            subplt[i][j].set_title(f"{actual} vs {pred}")
    plt.show()


if __name__ == "__main__":
    main()
