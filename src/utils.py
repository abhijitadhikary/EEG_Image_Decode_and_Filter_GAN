import argparse
import torch
import matplotlib.pyplot as plt

def get_args():
    args = argparse.Namespace()
    args.num_epochs = 3
    args.start_epoch = 0
    args.batch_size = True
    args.root = '..'
    args.dataset_folder = 'data'
    args.shuffle_train = True
    args.shuffle_test = True
    args.shuffle_validation = True
    args.batch_size_train = 192
    args.batch_size_test = 1
    args.batch_size_validation = 1
    args.loss_D_factor = 1
    args.learning_rate = 0.0002
    return args

def convert(source, min_value=0, max_value=1, type=torch.float32):
  smin = source.min()
  smax = source.max()

  a = (max_value - min_value) / (smax - smin)
  b = max_value - a * smax
  target = (a * source + b).astype(type)

  return target

def print_loss(loss_D, loss_G, index_epoch=-1, num_epochs=-1, mode='train'):
    if mode == 'test':
        print(f'\n---------> {mode}\t'
              f'D_loss_{mode}: {loss_D:.4f}\t'
              f'G_loss_{mode}: {loss_G:.4f}'
              )
    else:
        print(f'\n{mode}\tepoch: [{index_epoch + 1}/{num_epochs}]\t'
              f'D_loss_{mode}: {loss_D:.4f}\t'
              f'G_loss_{mode}: {loss_G:.4f}'
              )

def plot_train_vs_val_loss(loss_train, loss_val, mode='G'):
    plt.figure()
    type = 'Generator' if mode == 'G' else 'Discriminator'
    plt.title(f'{type} Loss')
    plt.plot(loss_train, label=f'{mode} loss train')
    plt.plot(loss_val, label=f'{mode} loss val')
    plt.legend()
    plt.show()