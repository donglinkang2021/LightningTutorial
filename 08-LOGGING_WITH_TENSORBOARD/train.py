import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from model import ImagenetTransferLearning
from dataset import CIFAR10DataModule
from callbacks import MyPrintingCallback, EarlyStopping
import config

torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    logger = TensorBoardLogger("lightning_logs", name="cifar10_resnet50_v0")

    dm = CIFAR10DataModule(
        data_dir=config.DATA_DIR, 
        batch_size=config.BATCH_SIZE, 
        num_workers=config.NUM_WORKERS
    )

    model = ImagenetTransferLearning(
        num_target_classes=config.NUM_CLASSES, 
        lr=config.LEARNING_RATE
    )
    # model = ImagenetTransferLearning.load_from_checkpoint(
    #     checkpoint_path=config.CHECKPOINT_PATH,
    #     num_target_classes=config.NUM_CLASSES,
    #     lr=config.LEARNING_RATE
    # )

    trainer = L.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        precision=config.PRECISION,
        logger=logger,
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