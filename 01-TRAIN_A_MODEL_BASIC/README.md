# TRAIN A MODEL (BASIC)

## Easy to understand, easy to implement.

```python
import os
import torch
from torch import nn
import torch.nn.functional as F
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

dataset = MNIST(
    os.getcwd(), 
    download=True, 
    transform=transforms.ToTensor()
)
train_loader = DataLoader(dataset)

# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())
```

## Train the model (a bit different)

```python
# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# train model
trainer = L.Trainer()
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
```

## Run it

> you can copy the following script to a file named `run.sh` and run it in the terminal

- run train.py

```bash
# run train.py code here 
# we use nohup to save the output to a file
# because printing to stdout will make the vscode terminal laggy

# define the path of the directory where the code is located
code_dir="01-TRAIN_A_MODEL_BASIC"

# you can use the following command to run the code
nohup python $code_dir/train.py > $code_dir/output.log &
# and then the code will be running in the background and save the output to output.log file
# you can close the terminal and the code will still be running
```

this will just output following in the terminal:

```bash
(GPT) root@asr:~/LightningTutorial# ./run.sh 
nohup: redirecting stderr to stdout
```

- see the `output.log` file

```bash
# to see the output, you can open the output.log file
# you can also use the following command to see the code running
python read_log.py --log_file $code_dir/output.log
```

- stop the running code

```bash
# if you want to stop the code, you can use the following command
ps -ef | grep train.py
# then you will see the process id (PID) of the code
```

such as:

```bash
(GPT) root@asr:~/LightningTutorial# ps -ef | grep train.py
root      390841       1 99 11:19 pts/20   00:01:13 python 01-TRAIN_A_MODEL_BASIC/train.py
root      391081   13543  0 11:20 pts/20   00:00:00 grep --color=auto train.py
```

```bash
# then you can use the following command to stop the code
kill -9 390841
```