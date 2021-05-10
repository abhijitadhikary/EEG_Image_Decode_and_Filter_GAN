import torch
args = 0
# dataloader = 0
# args.model_G = 0
# args.model_D = 0
# args.criterion_D = 0
# args.criterion_G = 0
# args.optimizer_D = 0
# args.optimizer_G = 0

# def train(args):
#     loss_D_train_running = []
#     loss_G_val_running = []
#     for index_epoch in range(args.start_epoch, args.num_epochs):
#         num_batches = len(dataloader)
#         for index_batch, batch in enumerate(dataloader):
#             loss_D_value, loss_G_value = forward_pass(args, dataloader, mode='train')
#
#             loss_G_val_running.append(loss_G_value)
#             loss_D_train_running.append(loss_D_value)
#
#             print(f'\nepoch: [{index_epoch + 1}/{args.num_epochs}]\t'
#                   f'batch: [{index_batch + 1}/{num_batches}]\t'
#                   f'D_loss: {loss_D_value:.4f}'
#                   f'G_loss: {loss_G_value:.4f}'
#                   )

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

        image_real, image_cat, identity, stimulus, alcoholism, condition_array_real, condition_array_fake = batch
        image_real, image_cat, identity, stimulus, alcoholism, condition_array_real, condition_array_fake = \
            image_real.to(args.device), image_cat.to(args.device), identity.to(args.device), stimulus.to(args.device), \
            alcoholism.to(args.device), condition_array_real.to(args.device), condition_array_fake.to(args.device)
        batch_size, num_channels, height, width = image_real.shape
        num_channels_cat = image_cat.shape[1]

        # generate image_fake image
        image_fake_temp = args.model_G.forward(image_cat)
        image_fake = torch.ones((batch_size, num_channels_cat, height, width)).to(args.device)
        image_fake[:, :3, :, :] = image_fake_temp
        image_fake[:, 3:, :, :] = image_cat[:, 3:, :, :]

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
        loss_G = args.criterion_G(out_D_fake, condition_array_fake)

        if mode == 'train':
            loss_G.backward()
            args.optimizer_G.step()

        loss_D_value = loss_D.detach().cpu().item()
        loss_G_value = loss_G.detach().cpu().item()

        # loss_D_value, loss_G_value = forward_pass(args, batch, mode='train')
        loss_D_epoch += loss_D_value
        loss_G_epoch += loss_G_value

    loss_D_epoch /= num_batches
    loss_G_epoch /= num_batches

    return loss_D_epoch, loss_G_epoch




