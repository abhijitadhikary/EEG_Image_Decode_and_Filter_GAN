import torch
import torchvision
import os

def forward_pass(args, dataloader, mode='train'):
    num_batches = len(dataloader)
    loss_D_epoch = 0
    loss_G_epoch = 0

    # train
    for index_batch, batch in enumerate(dataloader):
        if mode == 'train':
            args.model_D.train()
            args.model_G.train()
        else:
            args.model_D.eval()
            args.model_G.eval()

        image_real, image_conditioned, identity, stimulus, alcoholism, condition_array_real, condition_array_fake = batch
        image_real, image_conditioned, identity, stimulus, alcoholism, condition_array_real, condition_array_fake = \
            image_real.to(args.device), image_conditioned.to(args.device), identity.to(args.device), stimulus.to(args.device), \
            alcoholism.to(args.device), condition_array_real.to(args.device), condition_array_fake.to(args.device)
        batch_size, num_channels, height, width = image_real.shape
        num_channels_cat = image_conditioned.shape[1]

        # generate image_fake image
        image_fake_temp = args.model_G.forward(image_conditioned)
        image_fake = torch.ones((batch_size, num_channels_cat, height, width)).to(args.device)
        image_fake[:, :3, :, :] = image_fake_temp
        image_fake[:, 3:, :, :] = image_conditioned[:, 3:, :, :]

        # train discriminator - real
        args.model_D.zero_grad()
        out_D_real = args.model_D(image_real).squeeze(3)
        loss_D_real = args.criterion_D(out_D_real, condition_array_real)

        # train discriminator - fake
        out_D_fake = args.model_D(image_fake.detach()).squeeze(3)
        loss_D_fake = args.criterion_D(out_D_fake, condition_array_fake)

        loss_D = (loss_D_real + loss_D_fake) * args.loss_D_factor

        if mode == 'train':
            loss_D.backward()
            args.optimizer_D.step()

        # train generator
        out_D_fake = args.model_D(image_fake).squeeze(3)
        args.model_G.zero_grad()
        loss_G_gan = args.criterion_D(out_D_fake, condition_array_fake)
        loss_G_L1 = args.criterion_G(image_fake_temp, image_real[:, :3, :, :])
        loss_G = (loss_G_gan * args.loss_G_gan_factor) + (loss_G_L1 * args.loss_G_l1_factor)

        if mode == 'train':
            loss_G.backward()
            args.optimizer_G.step()

        loss_D_value = loss_D.detach().cpu().item()
        loss_G_value = loss_G.detach().cpu().item()

        # loss_D_value, loss_G_value = forward_pass(args, batch, mode='train')
        loss_D_epoch += loss_D_value
        loss_G_epoch += loss_G_value

        # create image grids for visualization
        if index_batch == 0:
            num_display = 8
            img_grid_real = torchvision.utils.make_grid(image_real[:num_display, :3, :, :], normalize=True, range=(0, 1))
            img_grid_fake = torchvision.utils.make_grid(image_fake_temp[:num_display], normalize=True, range=(0, 1))

            # combine the grids
            img_grid_combined = torch.hstack((img_grid_real, img_grid_fake))
            output_path = os.path.join('..', 'output', f'{args.index_epoch}_{index_batch}_{mode}.jpg')
            torchvision.utils.save_image(img_grid_combined, output_path)

    loss_D_epoch /= num_batches
    loss_G_epoch /= num_batches

    return loss_D_epoch, loss_G_epoch




