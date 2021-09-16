#!/usr/bin/env python3
"""
@Filename:    params.py.py
@Author:      dulanj
@Time:        2021-09-15 15.03
"""
import torch

CHANNELS = 3
NO_OF_CLASSES = 1
NO_OF_BOXES = 1
SPLIT_SIZE = 7
PATH = 'trained_models/best-model-parameters.pt'

LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
# LOAD_MODEL_FILE = 'trained_models/best-model-parameters.pt'
IMG_DIR = '/home/dulanj/Datasets/pizza/object_detection/dataset/images'
LABEL_DIR = '/home/dulanj/Datasets/pizza/object_detection/dataset/labels'
