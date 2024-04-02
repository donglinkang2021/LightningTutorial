# EARLY STOPPING

```python
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

...

# train model
early_stop_callback = EarlyStopping(monitor="val_loss", mode="min")
trainer = L.Trainer(max_epochs=20, callbacks=[early_stop_callback])
```

I think this is a good example of how to use the EarlyStopping callback.

## Refer

- [EARLY STOPPING](https://lightning.ai/docs/pytorch/stable/common/early_stopping.html)