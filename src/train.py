import torch
import torchvision
from dataset import get_dataloaders
from utils import get_args
import os

args = get_args()
print(args)

dataloader_train, dataloader_test, dataloader_validation = get_dataloaders(args)

batch = next(iter(dataloader_train))

