import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.strategies import DeepSpeedStrategy
from model import ImagenetTransferLearning
from dataset import CIFAR10DataModule
from callbacks import MyPrintingCallback, EarlyStopping
import config

torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    logger = TensorBoardLogger(
        save_dir="lightning_logs", 
        name="cifar10_resnet50_v2"
    )
    strategy = DeepSpeedStrategy()
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("lightning_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20)
    )

    dm = CIFAR10DataModule(
        data_dir=config.DATA_DIR, 
        batch_size=config.BATCH_SIZE, 
        num_workers=config.NUM_WORKERS
    )

    model = ImagenetTransferLearning(
        num_target_classes=config.NUM_CLASSES, 
        num_hidden=config.NUM_HIDDEN,
        lr=config.LEARNING_RATE
    )

    trainer = L.Trainer(
        logger=logger,
        profiler=profiler,
        strategy=strategy,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        precision=config.PRECISION,
        num_nodes=config.NUM_NODES,
        max_epochs=config.NUM_EPOCHS,
        callbacks=[
            MyPrintingCallback(), 
            EarlyStopping(monitor='val_loss')
        ]
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)

"""output

"""