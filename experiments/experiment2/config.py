import torch.nn as nn
from models.googlenet import model
from datasets.baseline_dataset import Dataset
from preprocessing.baseline_pipeline import transformer

SEED = 42
RESHAPE_SIZE = (48, 48)
PIPELINE_TRAIN = transformer
PIPELINE_VAL = None
DATASET = Dataset
BATCH_SIZE = 8
NUM_WORKERS = 6
CRITERION = nn.BCEWithLogitsLoss()
MODEL = model
OPTIMIZER = ''
MAX_EPOCHS = 40