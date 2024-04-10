# Training hyperparameters
NUM_CLASSES = 10
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
NUM_EPOCHS = 20
CHECKPOINT_PATH = 'lightning_logs/version_18/checkpoints/epoch=19-step=3140.ckpt'

# Dataset
DATA_DIR = 'data'
NUM_WORKERS = 4

# Compute related
ACCELERATOR = 'gpu'
DEVICES = [0, 1]
PRECISION = 32
NUM_NODES = 1
