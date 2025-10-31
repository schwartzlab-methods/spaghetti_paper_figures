import os
import torchvision.transforms.v2 as v2
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import random
from torchvision.utils import save_image
from skimage import color
import matplotlib.pyplot as plt
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def prep_datasets(datasets, indices=None):
    '''
    Seperates the datasets to two groups, either 80/20 randomly or by the indicies specified
    '''
    if indices:
        assert len(indices) == 2, "Indicies must be 2 lists of lists of Int"
        train_dataset = torch.utils.data.Subset(datasets, indices[0])
        val_dataset = torch.utils.data.Subset(datasets, indices[1])
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(datasets, [0.8,0.2])
    return train_dataset, val_dataset

def loader_data(dataset, weight_per_sample = None, batch_size = 16, shuffle: bool = False) -> torch.utils.data.DataLoader:
    '''
    Load the data
    '''
    # fix class imbalance by using weighted random sampler
    if weight_per_sample is not None:
        weights = torch.DoubleTensor(weight_per_sample)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler = sampler, num_workers=4)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def find_checkpoint(dir: str):
    '''
    Find the latest checkpoint in the directory
    '''
    files = []
    for path, _, file in os.walk(dir):
        for f in file:
            if f.endswith(".ckpt"):
                files.append(os.path.join(path, f))
    if len(files) == 0:
        return None
    else:
        return max(files, key=os.path.getctime)

def image_transform(do_augmentation: bool, do_cropping: bool, 
                    original_size: int = 256, new_size: int = 256):
    '''
    Randomly transform the image from by composing several transformations
    '''
    transform = [v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
                ]
    if do_augmentation:
        transform.extend([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(180),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1),shear=0.1),
            v2.GaussianBlur(3, sigma=(0.1, 2.0)),
        ])
    if do_cropping:
        transform.append(v2.RandomCrop(size=original_size//4))
    
    transform.append(v2.Resize((new_size,new_size)))
    return v2.Compose(transform)

def map_labels(labels: list[int], label_dict: dict) -> list[str]:
    rev_dic = {v: k for k, v in label_dict.items()}
    return [rev_dic[i] for i in labels]

def plot_masks(img_L: list[np.ndarray], original_L: list[np.ndarray], save_dir, name):
    '''
    Plot the masks on the image
    '''
    if not os.path.exists(os.path.join(save_dir, name)):
        os.makedirs(os.path.join(save_dir, name))
    for idx, img in enumerate(img_L):
        save_name = os.path.join(save_dir, name, f"{name}_{idx}.png")
        img_list = []
        original = original_L[idx]
        try:
            for i, mask in enumerate(img):
                if i == 0:
                    img_list.append(mask)
                else:
                    current_mask = color.label2rgb(mask,img_list[0],alpha=0.5, bg_label=0, bg_color=None)
                    img_list.append(current_mask)
        except ValueError:
            print("Shape mismatch")
            print(f"img shape: {img.shape}, original shape: {original.shape}")
            continue
        # plot using matplotlib
        fig, ax = plt.subplots(2, len(img), figsize=(20, 20))
        for i, image in enumerate(original):
            ax[0, i].imshow(image)
            ax[0, i].axis("off")
        for i, image in enumerate(img_list):
            ax[1, i].imshow(image)
            ax[1, i].axis("off")
        plt.savefig(save_name)
        plt.close()

        # tensor_img = torch.from_numpy(np.moveaxis(np.stack(img_list, axis=0), -1, 1))
        # save_image(tensor_img, os.path.join(save_dir, name, f"{name}_{idx}.png"), 
        #                 nrow=1, normalize=True, value_range=(-1, 1))

def sample_imgs(data_loader, idx, num, save_dir, name, result_img=None, do_crop=False):
    '''
    Sample num batches of images from the dataloader to display
    '''
    if not os.path.exists(os.path.join(save_dir, name)):
        os.makedirs(os.path.join(save_dir, name))
    if len(idx) == 0:
        print(f"No indices for {name}")
        return None
    idx_sampled = random.choices(idx, k=num)
    for i in idx_sampled:
        img, _ = data_loader[i]
        if do_crop:
            crop = v2.CenterCrop((img.shape[-2]//2, img.shape[-1]//2))
            img = crop(img)
        transform = v2.Resize((256,256))
        if result_img is not None:
            res = torch.cat((transform(img),torch.unsqueeze(result_img[i,:], 0)), 0)
        else: 
            res = img
        save_image(res, os.path.join(save_dir, name, f"{name}_{i}.png"), 
                        nrow=1, normalize=True, value_range=(-1, 1))
