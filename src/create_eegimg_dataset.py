import numpy as np
from scipy.io import loadmat
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def normalize_image(image):
    image = np.copy(image)
    image = ((image - image.min()) / (image.max() - image.min())) * 255
    return image

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Directory created: {path}')
    else:
        print(f'Directory already exists: {path}')

def create_dataset(variant=''):
    mat_dir = os.path.join('../data', 'mat')
    create_directory(mat_dir)
    mat_filename = f'uci_eeg_images_{variant}_within.mat'
    mat_path = os.path.join(mat_dir, mat_filename)
    mat = loadmat(mat_path)
    
    # extract features
    images = mat['data']
    label_alcoholism = mat['label_alcoholism']
    label_stimulus = mat['label_stimulus']
    label_id = mat['label_id']
    num_samples = len(label_id)
    
    # create dataframe
    all_labels = np.hstack([np.arange(num_samples).reshape(-1, 1), label_alcoholism, label_stimulus, label_id])
    column_names = ['filename', 'alcoholism', 'stimulus', 'id']
    df = pd.DataFrame(all_labels, columns=column_names)
    
    filenames = [f'{variant}_{index}.png' for index in range(num_samples)]
    df['filename'] = filenames
    
    # save dataframe
    save_dir_df = os.path.join('../data', 'df')
    create_directory(save_dir_df)
    df_filename = f'{variant}.csv'
    df_path = os.path.join(save_dir_df, df_filename)
    df.to_csv(df_path)
    
    # save images as numpy array
    save_dir_np = os.path.join('../data', 'numpy')
    create_directory(save_dir_np)
    np_filename = f'{variant}.npy'
    np_path = os.path.join(save_dir_np, np_filename)
    np.save(np_path, images)
    
    # save images as png
    save_dir_image = os.path.join('../data', 'images', variant)
    create_directory(save_dir_image)
        
    for index, image in tqdm(enumerate(images), total=len(images)):
        image = normalize_image(image)
        current_filename = f'{variant}_{index}.png'
        current_image_path = os.path.join(save_dir_image, current_filename)
        cv2.imwrite(current_image_path, image)
        
if __name__=='__main__':
    variants = ['train', 'test', 'validation']
    [create_dataset(variant) for variant in variants]