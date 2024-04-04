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
    
class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def prepare_data(self):
        # single gpu
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)
        
    
    def setup(self, stage=None):
        # multi-gpu
        entire_dataset = CIFAR10(root=self.data_dir, train=True, transform=self.transform, download=False)
        self.test_set = CIFAR10(root=self.data_dir, train=False, transform=self.transform, download=False)
        train_set_size = int(len(entire_dataset) * 0.8)
        valid_set_size = len(entire_dataset) - train_set_size
        seed = torch.Generator().manual_seed(1337)
        self.train_set, self.valid_set = data.random_split(entire_dataset, [train_set_size, valid_set_size], generator=seed)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_set, 
            self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False
        )
    

batch_size = 128
num_workers = 4
dm = CIFAR10DataModule(data_dir="data", batch_size=batch_size, num_workers=num_workers)

# model
prev_ckpt = "lightning_logs/version_12/checkpoints/epoch=19-step=3140.ckpt"
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
trainer.fit(model, dm)
trainer.validate(model, dm)
trainer.test(model, dm)

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