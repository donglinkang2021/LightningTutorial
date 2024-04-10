import torch
import lightning as L
from model import ImagenetTransferLearning
from dataset import CIFAR10DataModule
from callbacks import MyPrintingCallback, EarlyStopping
import config

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    # dataset
    dm = CIFAR10DataModule(
        data_dir=config.DATA_DIR, 
        batch_size=config.BATCH_SIZE, 
        num_workers=config.NUM_WORKERS
    )

    # model
    # model = ImagenetTransferLearning(
    #     num_target_classes=config.NUM_CLASSES, 
    #     lr=config.LEARNING_RATE
    # )
    model = ImagenetTransferLearning.load_from_checkpoint(
        checkpoint_path=config.CHECKPOINT_PATH,
        num_target_classes=config.NUM_CLASSES,
        lr=config.LEARNING_RATE
    )

    # training
    trainer = L.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        precision=config.PRECISION,
        num_nodes=config.NUM_NODES,
        max_epochs=config.NUM_EPOCHS,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor='val_loss')]
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)

"""output
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │    0.8140000104904175     │
│          test_f1          │    0.8140000104904175     │
│         test_loss         │    0.5401309132575989     │
└───────────────────────────┴───────────────────────────┘
"""