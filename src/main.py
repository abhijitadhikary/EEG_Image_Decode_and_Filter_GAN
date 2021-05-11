from utils import print_loss, plot_train_vs_val_loss, setup_model_parameters, save_model
from train import forward_pass

cuda_index = 0
args = setup_model_parameters()

for index_epoch in range(args.start_epoch, args.num_epochs):
    args.index_epoch = index_epoch
    # train
    loss_D_epoch_train, loss_G_epoch_train = forward_pass(args, args.dataloader_train, mode='train')
    args.loss_D_train_running.append(loss_D_epoch_train)
    args.loss_G_train_running.append(loss_G_epoch_train)
    print_loss(loss_D_epoch_train, loss_G_epoch_train, index_epoch, args.num_epochs, mode='train')
    # args.writer.add_scalar('Loss/train_D', loss_D_epoch_train, index_epoch+1)
    # args.writer.add_scalar('Loss/train_G', loss_G_epoch_train, index_epoch+1)

    # validate
    loss_D_epoch_val, loss_G_epoch_val = forward_pass(args, args.dataloader_val, mode='val')
    args.loss_D_val_running.append(loss_D_epoch_val)
    args.loss_G_val_running.append(loss_G_epoch_val)
    print_loss(loss_D_epoch_val, loss_G_epoch_val, index_epoch, args.num_epochs, mode='val')

    args.writer.add_scalars('Loss', {
        'train_D': loss_D_epoch_train,
        'val_D': loss_D_epoch_val,
        'train_G': loss_G_epoch_train,
        'val_G': loss_G_epoch_val
    }, index_epoch+1)
    # args.writer.add_scalars('Loss/val', {'D': loss_D_epoch_val, 'G': loss_G_epoch_val}, index_epoch + 1)
    # args.writer.add_scalar('Loss/val_D', loss_D_epoch_val, index_epoch + 1)
    # args.writer.add_scalar('Loss/val_G', loss_G_epoch_val, index_epoch + 1)

    save_model(args, loss_D_epoch_val, loss_G_epoch_val)
    # plot_train_vs_val_loss(args.loss_D_train_running, args.loss_D_val_running, mode='D')
    # plot_train_vs_val_loss(args.loss_G_train_running, args.loss_G_val_running, mode='G')

# test
loss_D_epoch_test, loss_G_epoch_test = forward_pass(args, args.dataloader_test, mode='test')
print_loss(loss_D_epoch_test, loss_G_epoch_test, mode='test')



