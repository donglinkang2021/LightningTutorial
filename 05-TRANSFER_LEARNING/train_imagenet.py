import os
import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import lightning as L
# from lightning.pytorch.strategies import DDPStrategy
import torchmetrics
from torchmetrics import Metric

torch.set_float32_matmul_precision('high')

class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total

class ImagenetTransferLearning(L.LightningModule):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        # self.accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes = num_target_classes)
        self.accuracy = Accuracy()
        self.f1_score = torchmetrics.F1Score(task = "multiclass", num_classes = num_target_classes)

    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x
    
    def training_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(logits, y)
        f1_score = self.f1_score(logits, y)
        self.log_dict({"train_loss": loss, "train_acc": accuracy, "train_f1": f1_score}, sync_dist=True,
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(logits, y)
        f1_score = self.f1_score(logits, y)
        self.log_dict({"val_loss": loss, "val_acc": accuracy, "val_f1": f1_score}, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(logits, y)
        f1_score = self.f1_score(logits, y)
        self.log_dict({"test_loss": loss, "test_acc": accuracy, "test_f1": f1_score}, sync_dist=True)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss, logits, y
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        return optimizer
    
# init data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_set = CIFAR10(root="data", train=True, transform=transform, download=False)
test_set = CIFAR10(root="data", train=False, transform=transform, download=False)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(1337)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

batch_size = 128
num_workers = 4
train_loader = DataLoader(train_set, batch_size, num_workers=num_workers)
valid_loader = DataLoader(valid_set, batch_size, num_workers=num_workers)
test_loader = DataLoader(test_set, batch_size, num_workers=num_workers)

# model
prev_ckpt = "lightning_logs/version_11/checkpoints/epoch=19-step=3140.ckpt"
model = ImagenetTransferLearning.load_from_checkpoint(prev_ckpt)

# training
trainer = L.Trainer(
    accelerator="gpu",
    # strategy=DDPStrategy(find_unused_parameters=True), # if you use with torch.no_grad() in the forward method
    devices=[0, 1],
    precision=16,
    num_nodes=1,
    max_epochs=20
)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
# test the model
trainer.test(model, dataloaders=test_loader)

"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │    0.8701000213623047     │
│         test_loss         │    0.37535327672958374    │
└───────────────────────────┴───────────────────────────┘

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │    0.8784000277519226     │
│         test_loss         │    0.3540230393409729     │
└───────────────────────────┴───────────────────────────┘
"""