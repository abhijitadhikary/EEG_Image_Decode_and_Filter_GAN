import argparse
import torch
import matplotlib.pyplot as plt
import os
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataloaders
from discriminator import Discriminator
from generator import UNet
import numpy as np


def get_args():
    args = argparse.Namespace()
    args.cuda_index = 0
    args.num_epochs = 20
    args.start_epoch = 0
    args.batch_size = True
    args.root = '..'
    args.dataset_folder = 'data'
    args.shuffle_train = True
    args.shuffle_test = True
    args.shuffle_validation = True
    args.batch_size_train = 192
    args.batch_size_test = 192
    args.batch_size_validation = 192
    args.loss_D_factor = 1
    args.learning_rate = 0.0002
    args.num_keep_best = 3
    args.resume_condition = True
    args.resume_epoch = 9
    args.checkpoint_path = os.path.join('..', 'experiments', 'checkpoints')
    return args

def setup_model_parameters():
    args = get_args()
    args.device = torch.device(f'cuda:{args.cuda_index}' if torch.cuda.is_available() else 'cpu')
    args.model_G = UNet().to(args.device)
    args.model_D = Discriminator().to(args.device)

    args.criterion_D = nn.BCELoss()
    args.criterion_G = nn.BCELoss()

    args.optimizer_D = optim.Adam(args.model_D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    args.optimizer_G = optim.Adam(args.model_G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    args.dataloader_train, args.dataloader_test, args.dataloader_val = get_dataloaders(args)

    args.writer_real = SummaryWriter(os.path.join('runs', 'real'))
    args.writer_fake = SummaryWriter(os.path.join('runs', 'fake'))
    create_dirs()

    args.loss_D_train_running = []
    args.loss_G_train_running = []
    args.loss_D_val_running = []
    args.loss_G_val_running = []

    args.loss_D_best = np.inf
    args.loss_G_best = np.inf

    # load model
    if args.resume_condition:
        load_model(args)

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
        print(f'---------> {mode}\t'
              f'D_loss_{mode}: {loss_D:.4f}\t'
              f'G_loss_{mode}: {loss_G:.4f}'
              )
    else:
        print(f'{mode}\tepoch: [{index_epoch + 1}/{num_epochs}]\t'
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

def create_dirs():
    dir_list = [
        ['..', 'data', 'df'],
        ['..', 'data', 'images', 'train', 'test', 'validation'],
        ['..', 'data', 'mat'],
        ['..', 'data', 'numpy'],
        ['..', 'notebooks'],
        ['..', 'output'],
        ['..', 'runs', 'real'],
        ['..', 'runs', 'fake'],
        ['..', 'experiments', 'checkpoints'],

    ]
    for current_dir in dir_list:
        current_path = current_dir[0]
        if len(current_dir) > 1:
            for sub_dir_index in range(1, len(current_dir)):
                current_path = os.path.join(current_path, current_dir[sub_dir_index])
        if not os.path.exists(current_path):
            os.makedirs(current_path)

def remove_all_files(args, path):
    all_files = os.listdir(path)
    if len(all_files) >= args.num_keep_best:
        current_full_path = os.path.join(path, all_files[0])
        os.remove(current_full_path)

def save_model(args, loss_Dl, loss_G):
    if loss_G < args.loss_G_best:
        args.loss_G_best = loss_G

        remove_all_files(args, args.checkpoint_path)

        save_path = os.path.join(args.checkpoint_path, f'{args.index_epoch+1}.pth')
        save_dict = {'epoch': args.index_epoch + 1,
                     'learning_rate': args.learning_rate,
                     'G_state_dict': args.model_G.state_dict(),
                     'G_optim_dict': args.optimizer_G.state_dict(),
                     'D_state_dict': args.model_D.state_dict(),
                     'D_optim_dict': args.optimizer_D.state_dict()
                     }

        torch.save(save_dict, save_path)
        print(f'New best model saved at {args.index_epoch+1}')

def load_model(args):

    load_path = os.path.join(args.checkpoint_path, f'{args.resume_epoch}.pth')

    if load_path is not None:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f'File {load_path} doesn\'t exist')

        checkpoint = torch.load(load_path)

        args.start_epoch = checkpoint['epoch']
        args.learning_rate = checkpoint['learning_rate']
        args.model_G.load_state_dict(checkpoint['G_state_dict'])
        args.model_D.load_state_dict(checkpoint['D_state_dict'])
        args.optimizer_G.load_state_dict(checkpoint['G_optim_dict'])
        args.optimizer_D.load_state_dict(checkpoint['D_optim_dict'])

        print(f'Model successfully loaded from epoch {args.resume_epoch}')

        return args