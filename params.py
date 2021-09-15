#!/usr/bin/env python3
"""
@Filename:    params.py.py
@Author:      dulanj
@Time:        2021-09-15 15.03
"""
import torch

CHANNELS=1
NO_OF_CLASSES = 5
NO_OF_BOXES = 1
SPLIT_SIZE=7
PATH='trained_models/best-model-parameters.pt'

LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = ''
IMG_DIR = 'data/screenshots'
LABEL_DIR = 'data/annotated_data'
