# Training hyperparameters
NUM_HIDDEN = 512
NUM_CLASSES = 10
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
NUM_EPOCHS = 20

# Dataset
DATA_DIR = 'data'
NUM_WORKERS = 4

# Compute related
ACCELERATOR = 'gpu'
DEVICES = [0, 1]
PRECISION = 32
NUM_NODES = 1
