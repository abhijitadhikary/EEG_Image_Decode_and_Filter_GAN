import argparse

def get_args():
    args = argparse.Namespace()
    args.batch_size = True
    args.root = '..'
    args.dataset_folder = 'data'
    args.shuffle_train = True
    args.shuffle_test = True
    args.shuffle_validation = True
    args.batch_size_train = 512
    args.batch_size_test = 1
    args.batch_size_validation = 1

    return args

def convert(source, min_value, max_value, type):
  smin = source.min()
  smax = source.max()

  a = (max_value - min_value) / (smax - smin)
  b = max_value - a * smax
  target = (a * source + b).astype(type)

  return target