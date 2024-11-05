import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

### Checking GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"RUNNING ON DEVICE: {device}")


### DATA LOADING
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

testing_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

def get_data(training_data, testing_data, batch_size):
    return (DataLoader(training_data, batch_size=batch_size, shuffle=True), DataLoader(testing_data, batch_size=batch_size, shuffle=True))


### NETWORK DEFINITIONS

class VariationalEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 240)

        self.fc_mu = nn.Linear(240, 10)
        self.fc_sigma = nn.Linear(240, 10)

        # distributions
        self.N = distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # GPU
        self.N.scale = self.N.scale.cuda()

        self.KL = 0

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_sigma(x))

        z = mu + sigma * self.N.sample(mu.shape) # reparameterisation trick
        self.KL = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        return z


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(10, 240),
            nn.ReLU(),
            nn.Linear(240, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x.reshape((-1, 1, 28, 28))


class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = VariationalEncoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


### TRAINING METHODS

def train(vae: VAE, dataloader: DataLoader, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)

    vae.train()

    for count, (X, _) in enumerate(dataloader):
        X = X.to(device)

        vae_pred = vae(X)

        loss = ((X - vae_pred)**2).sum() + vae.encoder.KL

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if(not(count % 100)):
            loss = loss.item()
            current = count * batch_size + len(X)
            print(f"Loss: {loss:>5f} [{current:>5d}/{size:>5d}]")


def test(vae: VAE, dataloader: DataLoader):
    num_batches = len(dataloader)

    vae.eval()

    test_loss = 0

    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)

            vae_pred = vae(X)
            test_loss += ((X - vae_pred)**2).sum() + vae.encoder.KL

    test_loss /= num_batches

    print(f"\nAverage loss: {test_loss:>5f}")


### MAIN FUNCTIONS

def main():

    # hyperparameters
    learning_rate = 1e-2
    batch_size = 64
    epochs = 25

    # loading data
    training_dataloader, testing_dataloader = get_data(training_data, testing_data, batch_size=batch_size)

    # instantiating model
    variational_autoencoder = VAE().to(device)

    # model attributes
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(variational_autoencoder.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n---------------")
        train(variational_autoencoder, training_dataloader, loss_fn, optimizer, batch_size)
        test(variational_autoencoder, testing_dataloader)
        print("\n\n")

        if not(t % 25):
            torch.save(variational_autoencoder.state_dict(), 'VAE_checkpoint.pth')

    torch.save(variational_autoencoder.state_dict(), 'VAE.pth')

    print("Job Finished.")

if __name__ == "__main__":
    main()




