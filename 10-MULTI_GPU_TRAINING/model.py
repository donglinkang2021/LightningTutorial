import torch
from torch import nn
import torchvision.models as models
import lightning as L
import torchmetrics
import torchvision

class ImagenetTransferLearning(L.LightningModule):
    def __init__(self, num_target_classes:int, num_hidden:int, lr:float):
        super().__init__()
        self.save_hyperparameters()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # use the pretrained model to classify cifar-10 (10 image classes)
        
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_target_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes = num_target_classes)
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
        
        if batch_idx % 100 == 0:
            x, _ = batch
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 3, 224, 224))
            self.logger.experiment.add_image('cifar10_images', grid, global_step=self.global_step)

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer