import torch.nn as nn
from models.custom_3 import model
from datasets.baseline_dataset import Dataset
from preprocessing.baseline_pipeline import transformer_train, transformer_val

"""
Test custom simple model with more layers
"""
SEED = 42
RESHAPE_SIZE = (48, 48)
PIPELINE_TRAIN = transformer_train
PIPELINE_VAL = transformer_val
DATASET = Dataset
BATCH_SIZE = 64
NUM_WORKERS = 10
CRITERION = nn.CrossEntropyLoss()
MODEL = model
OPTIMIZER = ''
MAX_EPOCHS = 20