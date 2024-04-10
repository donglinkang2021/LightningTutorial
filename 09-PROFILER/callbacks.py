from lightning.pytorch.callbacks import Callback, EarlyStopping

class MyPrintingCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Training is starting!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done!")