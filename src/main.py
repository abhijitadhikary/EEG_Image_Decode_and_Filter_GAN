import torch
import torchvision
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from utils import get_args, print_loss, plot_train_vs_val_loss
from generator import UNet
from discriminator import Discriminator
from train import forward_pass

cuda_index = 0

args = get_args()
args.device = torch.device(f'cuda:{cuda_index}' if torch.cuda.is_available() else 'cpu')
# print(args)
args.model_G = UNet().to(args.device)
args.model_D = Discriminator().to(args.device)

args.criterion_D = nn.BCELoss()
args.criterion_G = nn.BCELoss()

args.optimizer_D = optim.Adam(args.model_D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
args.optimizer_G = optim.Adam(args.model_G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

dataloader_train, dataloader_test, dataloader_val = get_dataloaders(args)

loss_D_train_running = []
loss_G_train_running = []
loss_D_val_running = []
loss_G_val_running = []

for index_epoch in range(args.start_epoch, args.num_epochs):
    # train
    loss_D_epoch_train, loss_G_epoch_train = forward_pass(args, dataloader_train, mode='train')
    loss_D_train_running.append(loss_D_epoch_train)
    loss_G_train_running.append(loss_G_epoch_train)
    print_loss(loss_D_epoch_train, loss_G_epoch_train, index_epoch, args.num_epochs, mode='train')

    # validate
    loss_D_epoch_val, loss_G_epoch_val = forward_pass(args, dataloader_val, mode='val')
    loss_D_val_running.append(loss_D_epoch_val)
    loss_G_val_running.append(loss_G_epoch_val)
    print_loss(loss_D_epoch_val, loss_G_epoch_val, index_epoch, args.num_epochs, mode='val')

# test
loss_D_epoch_test, loss_G_epoch_test = forward_pass(args, dataloader_test, mode='test')
print_loss(loss_D_epoch_test, loss_G_epoch_test, mode='test')


plot_train_vs_val_loss(loss_D_train_running, loss_D_val_running, mode='D')
plot_train_vs_val_loss(loss_G_train_running, loss_G_val_running, mode='G')

