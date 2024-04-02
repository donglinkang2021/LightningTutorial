import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L

# define the nn model
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(28 * 28, 64), 
            nn.ReLU(), 
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(3, 64), 
            nn.ReLU(), 
            nn.Linear(64, 28 * 28)
        )

    def forward(self, x):
        return self.l1(x)
    
# define lightning model
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Load data sets
transform = transforms.ToTensor()
train_set = MNIST(root="data", download=False, train=True, transform=transform)
test_set = MNIST(root="data", download=False, train=False, transform=transform)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(1337)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

batch_size=64
train_loader = DataLoader(train_set, batch_size)
valid_loader = DataLoader(valid_set, batch_size)
test_loader = DataLoader(test_set, batch_size)

# model
prev_ckpt = "lightning_logs/version_4/checkpoints/epoch=2-step=2250.ckpt"
model = LitAutoEncoder.load_from_checkpoint(prev_ckpt, encoder=Encoder(), decoder=Decoder())

# train model
trainer = L.Trainer(max_epochs=3)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
# test the model
trainer.test(model, dataloaders=test_loader)