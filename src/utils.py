import argparse
import torch

def get_args():
    args = argparse.Namespace()
    args.num_epochs = 5
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