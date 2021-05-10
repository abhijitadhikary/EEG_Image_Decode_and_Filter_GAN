import torch
import torchvision
import numpy as np
import os
import torch.nn as nn
from dataset import get_dataloaders
from utils import get_args
from generator import UNet
from discriminator import Discriminator

args = get_args()
# print(args)
model_G = UNet()
model_D = Discriminator()

criterion_D = nn.BCELoss()

dataloader_train, dataloader_test, dataloader_validation = get_dataloaders(args)

for index_epoch in range(args.start_epoch, args.num_epochs):
    num_batches = len(dataloader_train)
    for index_batch, batch in enumerate(dataloader_train):

        image_real, image_cat, identity, stimulus, alcoholism, condition_array_real, condition_array_fake = batch
        batch_size, num_channels, height, width = image_real.shape
        num_channels_cat = image_cat.shape[1]

        # train discriminator
        model_D.zero_grad()
        out_disc_real = model_D(image_real).squeeze(3)
        loss_disc_real = criterion_D(out_disc_real, condition_array_real)



        # generator
        fake_temp = model_G.forward(image_cat)
        fake = torch.ones((batch_size, num_channels_cat, height, width))
        fake[:, :3, :, :] = fake_temp
        fake[:, 3:, :, :] = torch.tensor(image_cat[:, 3:, :, :])

        # discriminator
        out_disc_fake = model_D(fake).squeeze(3)
        loss_disc = criterion_D(out_disc_fake, condition_array_fake)

        print()




        # print(f'\nepoch: [{index_epoch}/{args.num_epochs}]\t'
        #       f'batch: [{index_batch}/{num_batches}]\t'
        #       f'identity: {identity.data.item()}\t'
        #       f'stimulus: {stimulus.data.item()}\t'
        #       f'alcoholism: {alcoholism.data.item()}\t'
        #       f'conditions: {np.array(condition_array).reshape(-1)}\t'
        #       )
        break