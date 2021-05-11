import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from scipy.io import loadmat

class CreateDataset(Dataset):
    def __init__(self, args, variant, transform=None):
        dataset_path = os.path.join(args.root, args.dataset_folder)
        self.mat_path = os.path.join(dataset_path, 'mat', f'uci_eeg_images_{variant}_within.mat')

        mat = loadmat(self.mat_path)
        self.identity = mat['label_id']
        self.stimulus = mat['label_stimulus']
        self.alcoholism = mat['label_alcoholism']

        self.images = mat['data']

        self.num_samples = len(self.images)

        self.transform = transform
        self.toTensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        identity = torch.tensor(self.identity[index], dtype=torch.int64)
        stimulus = torch.tensor(self.stimulus[index], dtype=torch.int64)
        alcoholism = torch.tensor(self.alcoholism[index], dtype=torch.int64)

        image = self.images[index]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        image = convert(image, 0, 1, torch.float32)

        if not self.transform is None:
            image = self.transform(image)

        # prepare mask for conditional generation
        image_conditioned, condition_array_fake, condition_array_real = get_conditioned_image(image)
        image_real = create_real_condiitoned_image(image)

        return image_real, image_conditioned, identity, stimulus, alcoholism, condition_array_real, condition_array_fake


def get_dataloaders(args):

    dataset_train = CreateDataset(args, 'train')
    dataset_test = CreateDataset(args, 'test')
    dataset_validation = CreateDataset(args, 'validation')

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size_train, shuffle=args.shuffle_train)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=args.shuffle_test)
    dataloader_validation = DataLoader(dataset_validation, batch_size=args.batch_size_validation, shuffle=args.shuffle_validation)

    return dataloader_train, dataloader_test, dataloader_validation

def get_conditioned_image(image):
    '''
    create random conditions {0,1} for each feature and concatenate them to the image
    '''
    num_channels, height, width = image.shape
    num_features = 3
    condition_array_fake = torch.tensor([(np.random.rand(1) > 0.5).astype(np.float32) for index in range(num_features)])
    filter_identity, filter_stimulus, filter_alcoholism = ([condition_array_fake[index] * torch.ones((1, height, width), dtype=torch.float32) for index in range(3)])
    image_conditioned = torch.cat((image, filter_identity, filter_stimulus, filter_alcoholism), dim=0)
    condition_array_real = torch.ones_like(condition_array_fake)
    return image_conditioned, condition_array_fake, condition_array_real

def create_real_condiitoned_image(image):
    num_channels, height, width = image.shape
    ones = torch.ones((1, height, width), dtype=torch.float32)
    image_real = torch.cat((image, ones, ones, ones), dim=0)
    return image_real

def convert(source, min_value=0, max_value=1, type=torch.float32):
  smin = source.min()
  smax = source.max()

  a = (max_value - min_value) / (smax - smin)
  b = max_value - a * smax
  target = (a * source + b)

  return target