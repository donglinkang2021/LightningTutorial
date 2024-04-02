import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)

# setup data
dataset = MNIST("data", download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

"""output
(GPT) root@asr:~/LightningTutorial# python dev.py 
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to /root/LightningTutorial/MNIST/raw/train-images-idx3-ubyte.gz
100%|████████████████████████████████████████| 9912422/9912422 [00:06<00:00, 1540877.77it/s]
Extracting /root/LightningTutorial/MNIST/raw/train-images-idx3-ubyte.gz to /root/LightningTutorial/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to /root/LightningTutorial/MNIST/raw/train-labels-idx1-ubyte.gz
100%|████████████████████████████████████████████| 28881/28881 [00:00<00:00, 1298401.79it/s]
Extracting /root/LightningTutorial/MNIST/raw/train-labels-idx1-ubyte.gz to /root/LightningTutorial/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to /root/LightningTutorial/MNIST/raw/t10k-images-idx3-ubyte.gz
100%|█████████████████████████████████████████| 1648877/1648877 [00:02<00:00, 555246.21it/s]
Extracting /root/LightningTutorial/MNIST/raw/t10k-images-idx3-ubyte.gz to /root/LightningTutorial/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to /root/LightningTutorial/MNIST/raw/t10k-labels-idx1-ubyte.gz
100%|█████████████████████████████████████████████| 4542/4542 [00:00<00:00, 14388616.89it/s]
Extracting /root/LightningTutorial/MNIST/raw/t10k-labels-idx1-ubyte.gz to /root/LightningTutorial/MNIST/raw

GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
You are using a CUDA device ('NVIDIA GeForce RTX 4090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
Missing logger folder: /root/LightningTutorial/lightning_logs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name    | Type       | Params
---------------------------------------
0 | encoder | Sequential | 50.4 K
1 | decoder | Sequential | 51.2 K
---------------------------------------
101 K     Trainable params
0         Non-trainable params
101 K     Total params
0.407     Total estimated model params size (MB)
/opt/data/private/linkdom/miniconda3/envs/GPT/lib/python3.9/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:441: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=63` in the `DataLoader` to improve performance.
Epoch 0: 100%|███████████████████████████████████| 100/100 [00:01<00:00, 52.63it/s, v_num=0]`Trainer.fit` stopped: `max_epochs=1` reached.
Epoch 0: 100%|███████████████████████████████████| 100/100 [00:01<00:00, 52.45it/s, v_num=0]
"""