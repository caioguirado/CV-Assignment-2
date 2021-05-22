import torch.nn as nn
from models.googlenet import model
from datasets.baseline_dataset import Dataset
from preprocessing.baseline_pipeline import transformer_train, transformer_val

SEED = 42
PIPELINE_TRAIN = transformer_train
PIPELINE_VAL = transformer_val
DATASET = Dataset
BATCH_SIZE = 64
NUM_WORKERS = 6
CRITERION = nn.CrossEntropyLoss()
MODEL = model
OPTIMIZER = ''
MAX_EPOCHS = 10