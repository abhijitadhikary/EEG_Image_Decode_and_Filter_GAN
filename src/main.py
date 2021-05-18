from utils import print_loss, plot_train_vs_val_loss, setup_model_parameters, save_model
from train import forward_pass

args = setup_model_parameters()

for index_epoch in range(args.start_epoch, args.num_epochs):
    args.index_epoch = index_epoch
    # train
    loss_D_cls_epoch_train, loss_D_adv_epoch_train, loss_D_total_epoch_train, loss_G_epoch_train = forward_pass(args, args.dataloader_train, mode='train')
    args.loss_D_cls_train_running.append(loss_D_cls_epoch_train)
    args.loss_D_adv_train_running.append(loss_D_adv_epoch_train)
    args.loss_D_total_train_running.append(loss_D_total_epoch_train)
    args.loss_G_train_running.append(loss_G_epoch_train)
    print_loss(loss_D_cls_epoch_train, loss_D_adv_epoch_train, loss_D_total_epoch_train, loss_G_epoch_train, index_epoch, args.num_epochs, mode='train')

    # validate
    loss_D_cls_epoch_val, loss_D_adv_epoch_val, loss_D_total_epoch_val, loss_G_epoch_val = forward_pass(args, args.dataloader_val, mode='val')
    args.loss_D_cls_val_running.append(loss_D_cls_epoch_val)
    args.loss_D_adv_val_running.append(loss_D_adv_epoch_val)
    args.loss_D_total_val_running.append(loss_D_total_epoch_val)
    args.loss_G_val_running.append(loss_G_epoch_val)
    print_loss(loss_D_cls_epoch_val, loss_D_adv_epoch_val, loss_D_total_epoch_val, loss_G_epoch_val, index_epoch, args.num_epochs, mode='val')

    args.writer.add_scalars('Loss', {
        'train_D_cls': loss_D_cls_epoch_train,
        'train_D_adv': loss_D_adv_epoch_train,
        'val_D_cls': loss_D_cls_epoch_val,
        'val_D_adv': loss_D_adv_epoch_val,
        'train_G': loss_G_epoch_train,
        'val_G': loss_G_epoch_val
    }, index_epoch+1)

    save_model(args, loss_D_cls_epoch_val, loss_D_adv_epoch_val, loss_G_epoch_val) # need to update this function to accommodate loss_D_adv

# test
loss_D_cls_epoch_test, loss_D_adv_epoch_test, loss_D_total_epoch_test, loss_G_epoch_test = forward_pass(args, args.dataloader_test, mode='test')
print_loss(loss_D_cls_epoch_test, loss_D_adv_epoch_test, loss_D_total_epoch_test, loss_G_epoch_test, mode='test')