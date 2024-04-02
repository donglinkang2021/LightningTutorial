# VALIDATE AND TEST A MODEL (BASIC)

## Add validation and testing to your model training pipeline.

```python
class LitAutoEncoder(L.LightningModule):
        ...

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
```

## Spilt the data into training, validation, and test sets.

```python
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
```

## Run the validation and test steps.

```python
# model
model = LitAutoEncoder(Encoder(), Decoder())

# train model
trainer = L.Trainer(max_epochs=3)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
# test the model
trainer.test(model, dataloaders=test_loader)
```

## Run the script

```bash
code_dir="02-VALIDATE_AND_TEST_A_MODEL_BASIC"

# nohup python $code_dir/train.py > $code_dir/output.log &

python read_log.py --log_file $code_dir/output.log
```

## Results

```bash
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
You are using a CUDA device ('NVIDIA GeForce RTX 4090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name    | Type    | Params
------------------------------------
0 | encoder | Encoder | 50.4 K
1 | decoder | Decoder | 51.2 K
------------------------------------
101 K     Trainable params
0         Non-trainable params
101 K     Total params
0.407     Total estimated model params size (MB)

Testing DataLoader 0: 100%|██████████| 157/157 [00:00<00:00, 207.95it/s]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_loss         │    0.04158072918653488    │
└───────────────────────────┴───────────────────────────┘
```

## Refer

- [VALIDATE AND TEST A MODEL (BASIC)](https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html)