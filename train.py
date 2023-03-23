import torch
from torch.utils.data import DataLoader, random_split
from dataset import CIFARDataset
from network import NeuralNetwork
import config
from torch import optim, nn
import os



def main() -> None:

    dataset: CIFARDataset = CIFARDataset()

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, (train_size, test_size))

    train_data_loader: DataLoader = DataLoader(train_dataset, batch_size=config.batch_size)
    test_data_loader: DataLoader = DataLoader(test_dataset, batch_size=1000)

    if os.path.isfile("trained_model.pt"):
        model: NeuralNetwork = torch.load("trained_model.pt")
    else:
        model: NeuralNetwork = NeuralNetwork().to(config.device)

    print(sum(param.numel() for param in model.parameters()))
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.SGD = optim.SGD(model.parameters(), lr=1e-4)
    
    for epoch in range(config.num_epochs + 1):

        for batch, labels in train_data_loader:

            output: torch.Tensor = model(batch)

            loss: torch.Tensor = criterion(output, labels)

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

        print(f"Epoch {epoch}")
    
    with torch.no_grad():
        test, labels = next(iter(test_data_loader))
        out = torch.argmax(model(test), dim=1)
        print(f"Accuracy: {100 * torch.mean(out == labels, dtype=torch.float32).item():.2f}")

    torch.save(model, 'trained_model.pt')


if __name__ == "__main__":
    main()