import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

### Checking GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"RUNNING ON DEVICE: {device}")


### LOADING DATASETS

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# class WrappedDataLoader:
#     def __init__(self, dataloader, func):
#         self.dataloader = dataloader
#         self.func = func

#     def __len__(self):
#         return len(self.dataloader)
    
#     def __iter__(self):
#         for b in self.dataloader:
#             yield(self.func(*b))

def get_data(training_data, test_data, batch_size):
    return (DataLoader(training_data, batch_size=batch_size, shuffle=True), DataLoader(test_data, batch_size=batch_size, shuffle=True))

# def gpu_preprocess(x, y):
#     # x is input, y is labels - sending to GPU
#     return (x.view(-1, 1, 28, 28).to(device), y.to(device))


### NEURAL NETWORK DEFINITIONS

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 240),
            nn.ReLU(),
            nn.Linear(240, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 30)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(30, 60),
            nn.ReLU(),
            nn.Linear(60, 120),
            nn.ReLU(),
            nn.Linear(120, 240),
            nn.ReLU(),
            nn.Linear(240, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x




class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


### TRAINING AND TESTING LOOPS

def ae_train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)

    # set models to training mode
    model.train()

    for count, (X, _) in enumerate(dataloader):
        X = X.to(device) # GPU

        ae_pred = model(X)
        loss = loss_fn(ae_pred, nn.Flatten()(X))

        # bp step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if(not(count % 100)):
            loss = loss.item()
            current = count * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def ae_test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    test_loss = 0

    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device) #gpu

            ae_pred = model(X)
            test_loss += loss_fn(ae_pred, nn.Flatten()(X)).item()

    test_loss /= num_batches

    print(f"Avg loss: {test_loss:>8f}\n")


### MAIN FUNCTION

def main():

    # hyperparameters
    learning_rate = 1e-2
    batch_size = 64
    epochs = 250

    # loading data
    train_dataloader, test_dataloader = get_data(training_data, test_data, batch_size)

    # instantiating model
    autoencoder_model = Autoencoder().to(device)

    ae_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------")
        ae_train_loop(train_dataloader, model=autoencoder_model, loss_fn=ae_loss_fn, optimizer=optimizer, batch_size=batch_size)
        ae_test_loop(test_dataloader, autoencoder_model, ae_loss_fn)

    torch.save(autoencoder_model.state_dict(), 'AE.pth') # save model

    print("Job finished.")


if __name__ == "__main__":
    main()