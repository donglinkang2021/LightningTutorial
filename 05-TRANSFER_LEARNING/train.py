import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L

prev_ckpt = "lightning_logs/version_6/checkpoints/epoch=19-step=15000.ckpt"

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
class AutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class MNIST10Classifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        # init the pretrained LightningModule
        self.feature_extractor = AutoEncoder.load_from_checkpoint(prev_ckpt)
        self.feature_extractor.freeze()

        # the autoencoder outputs a 100-dim representation and CIFAR-10 has 10 classes
        self.classifier = nn.Linear(28 * 28, 10)

    def forward(self, x):
        representations = self.feature_extractor(x)
        x = self.classifier(representations)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)

        # log the accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("test_loss", loss)

        # log the accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc)
    
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
model = MNIST10Classifier()

# train model
trainer = L.Trainer(max_epochs=20, default_root_dir="lightning_logs/CiFar10Classifier")
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
# test the model
trainer.test(model, dataloaders=test_loader)

"""output
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │    0.7973999977111816     │
│         test_loss         │    0.5767496824264526     │
└───────────────────────────┴───────────────────────────┘
"""