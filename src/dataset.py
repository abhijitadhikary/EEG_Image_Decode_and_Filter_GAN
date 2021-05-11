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
        self.identity = convert(mat['label_id'], 0, 1)
        self.stimulus = convert(mat['label_stimulus'], 0, 1)
        self.alcoholism = convert(mat['label_alcoholism'], 0, 1)

        self.images = convert(mat['data'], 0, 1)

        self.num_samples = len(self.images)

        self.transform = transform
        self.toTensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        identity = torch.tensor(self.identity[index], dtype=torch.float32)
        stimulus = torch.tensor(self.stimulus[index], dtype=torch.float32)
        alcoholism = torch.tensor(self.alcoholism[index], dtype=torch.float32)

        image = self.images[index]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        image = convert(image, 0, 1) ################################################################################################################

        if not self.transform is None:
            image = self.transform(image)

        # prepare mask for conditional generation
        image_c_real, image_c_fake, condition_array_real, condition_array_fake = get_conditioned_image(image)
        targets_real = torch.cat((identity, stimulus, alcoholism)).reshape(-1, 1)
        targets_fake = targets_real * condition_array_fake # float
        return image, image_c_real, image_c_fake, condition_array_real, condition_array_fake, identity, stimulus, alcoholism, targets_real, targets_fake


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
    image_c_fake = torch.cat((image, filter_identity, filter_stimulus, filter_alcoholism), dim=0)

    condition_array_real = torch.ones_like(condition_array_fake)
    filter_identity, filter_stimulus, filter_alcoholism = ([condition_array_fake[index] * torch.ones((1, height, width), dtype=torch.float32) for index in range(3)])
    image_c_real = torch.cat((image, filter_identity, filter_stimulus, filter_alcoholism), dim=0)

    return image_c_real, image_c_fake, condition_array_real, condition_array_fake

def convert(source, min_value=0, max_value=1):
  smin = source.min()
  smax = source.max()

  a = (max_value - min_value) / (smax - smin)
  b = max_value - a * smax
  target = (a * source + b)

  return target