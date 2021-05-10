import torch
import torchvision
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from utils import get_args
from generator import UNet
from discriminator import Discriminator

cuda_index = 0
device = torch.device(f'cuda:{cuda_index}' if torch.cuda.is_available() else 'cpu')

args = get_args()
# print(args)
model_G = UNet().to(device)
model_D = Discriminator().to(device)

criterion_D = nn.BCELoss()
criterion_G = nn.BCELoss()

optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
optimizer_G = optim.Adam(model_G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

dataloader_train, dataloader_test, dataloader_validation = get_dataloaders(args)

loss_D_running = []
loss_G_running = []

for index_epoch in range(args.start_epoch, args.num_epochs):
    num_batches = len(dataloader_train)
    for index_batch, batch in enumerate(dataloader_train):

        image_real, image_cat, identity, stimulus, alcoholism, condition_array_real, condition_array_fake = batch
        image_real, image_cat, identity, stimulus, alcoholism, condition_array_real, condition_array_fake = image_real.to(device), image_cat.to(device), identity.to(device), stimulus.to(device), alcoholism.to(device), condition_array_real.to(device), condition_array_fake.to(device)
        batch_size, num_channels, height, width = image_real.shape
        num_channels_cat = image_cat.shape[1]

        # generate image_fake image
        image_fake_temp = model_G.forward(image_cat)
        image_fake = torch.ones((batch_size, num_channels_cat, height, width)).to(device)
        image_fake[:, :3, :, :] = image_fake_temp
        image_fake[:, 3:, :, :] = image_cat[:, 3:, :, :]

        # train discriminator - real
        model_D.zero_grad()
        out_D_real = model_D(image_real).squeeze(3)
        loss_D_real = criterion_D(out_D_real, condition_array_real)

        # train discriminator - fake
        out_D_fake = model_D(image_fake.detach()).squeeze(3)
        loss_D_fake = criterion_D(out_D_fake, condition_array_fake)

        loss_D = (loss_D_real + loss_D_fake) * args.loss_D_factor
        loss_D.backward()
        optimizer_D.step()



        # train generator
        out_D_fake = model_D(image_fake).squeeze(3)
        model_G.zero_grad()
        loss_G = criterion_G(out_D_fake, condition_array_fake)
        loss_G.backward()
        optimizer_G.step()

        loss_D_value = loss_D.detach().cpu().item()
        loss_G_value = loss_G.detach().cpu().item()

        print(f'\nepoch: [{index_epoch+1}/{args.num_epochs}]\t'
              f'batch: [{index_batch+1}/{num_batches}]\t'
              f'D_loss: {loss_D_value:.4f}'
              f'G_loss: {loss_G_value:.4f}'
              )
        loss_G_running.append(loss_G_value)
        loss_D_running.append(loss_D_value)


        # print(f'\nepoch: [{index_epoch}/{args.num_epochs}]\t'
        #       f'batch: [{index_batch}/{num_batches}]\t'
        #       f'identity: {identity.data.item()}\t'
        #       f'stimulus: {stimulus.data.item()}\t'
        #       f'alcoholism: {alcoholism.data.item()}\t'
        #       f'conditions: {np.array(condition_array).reshape(-1)}\t'
        #       )

plt.figure()
plt.plot(loss_D_running, label='D loss')
plt.plot(loss_G_running, label='G loss')
plt.show()