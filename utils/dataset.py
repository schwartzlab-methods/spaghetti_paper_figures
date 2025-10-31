import random
from PIL import Image
import os
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
import torch
import numpy as np
import tifffile

class CycleGANDataset(Dataset):
    def __init__(self, path_1: str, path_2: list[str], 
                 transform, size=256, num_sample=None,
                 do_crop_1=False, do_augmentation=False, keep_RGB_img=False):
        super(CycleGANDataset).__init__()
        # image paths
        random.seed(42)

        domain1_images = [os.path.join(path, x) 
                               for path, _, files in os.walk(path_1) for x in files]
        
        if num_sample:
            try:
                self.domain1_images = random.sample(domain1_images, k=num_sample*len(path_2))
            except ValueError:
                self.domain1_images = random.choices(domain1_images, k=num_sample*len(path_2))
        else:
            self.domain1_images = domain1_images
        
        self.domain2_images = []
        for each in path_2:
            domain2_images = [os.path.join(path, x) 
                               for path, _, files in os.walk(each) for x in files]
            if num_sample:
                try:
                    domain2_images_sampled = random.sample(domain2_images, k=num_sample)
                except ValueError:
                    domain2_images_sampled = random.choices(domain2_images, k=num_sample)
                self.domain2_images.extend(domain2_images_sampled)
            else:
                self.domain2_images.extend(domain2_images)

        # others
        self.length_dataset = max(len(self.domain1_images), len(self.domain2_images))
        self.domain1_len = len(self.domain1_images)
        self.domain2_len = len(self.domain2_images)
        self.transform_1 = transform(new_size=size, do_augmentation=do_augmentation, do_cropping=do_crop_1)
        self.transform_2 = transform(new_size=size, do_augmentation=do_augmentation, do_cropping=False)
        self.keep_RGB = keep_RGB_img

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        domain1_img_path = self.domain1_images[index % self.domain1_len]
        domain2_img_path = self.domain2_images[index % self.domain2_len]

        img_type = "RGB" if self.keep_RGB else "L" #L for gray, RBG for RGB

        domain1_img = Image.open(domain1_img_path).convert(img_type)
        domain2_img = Image.open(domain2_img_path).convert(img_type) 

        domain1_img = self.transform_1(domain1_img)
        domain2_img = self.transform_2(domain2_img)

        return domain1_img, domain2_img

class CycleGANDatasetForGFP(Dataset):
    def __init__(self, path_1: str, path_2: list[str], 
                 transform, size=256, num_sample=None,
                 do_crop_1=False, do_augmentation=False, keep_RGB_img=False):
        super(CycleGANDataset).__init__()
        # image paths
        random.seed(42)

        domain1_images = [os.path.join(path, x) 
                               for path, _, files in os.walk(path_1) for x in files if x.endswith("0.jpg")]
        
        if num_sample:
            try:
                self.domain1_images = random.sample(domain1_images, k=num_sample*len(path_2))
            except ValueError:
                self.domain1_images = random.choices(domain1_images, k=num_sample*len(path_2))
        else:
            self.domain1_images = domain1_images
        
        self.domain2_images = []
        for each in path_2:
            domain2_images = [os.path.join(path, x) 
                               for path, _, files in os.walk(each) for x in files]
            if num_sample:
                try:
                    domain2_images_sampled = random.sample(domain2_images, k=num_sample)
                except ValueError:
                    domain2_images_sampled = random.choices(domain2_images, k=num_sample)
                self.domain2_images.extend(domain2_images_sampled)
            else:
                self.domain2_images.extend(domain2_images)

        # others
        self.length_dataset = max(len(self.domain1_images), len(self.domain2_images))
        self.domain1_len = len(self.domain1_images)
        self.domain2_len = len(self.domain2_images)
        self.transform_1 = transform(new_size=size, do_augmentation=do_augmentation, do_cropping=do_crop_1)
        self.transform_2 = transform(new_size=size, do_augmentation=do_augmentation, do_cropping=False)
        self.keep_RGB = keep_RGB_img

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        domain1_img_path = self.domain1_images[index % self.domain1_len]
        domain2_img_path = self.domain2_images[index % self.domain2_len]

        img_type = "RGB" if self.keep_RGB else "L" #L for gray, RBG for RGB

        domain1_img = Image.open(domain1_img_path).convert(img_type)
        domain2_img = Image.open(domain2_img_path).convert(img_type) 

        domain1_img = self.transform_1(domain1_img)
        domain2_img = self.transform_2(domain2_img)

        return domain1_img, domain2_img

class CustomImageDataset(Dataset):
    def __init__(self, paths, transform=None, sample_per_class=None, if_classification=False):
        super(CustomImageDataset, self).__init__()
        self.paths = paths
        self.transform = transform
        self.sample_per_class = sample_per_class
        self.images = []
        self.classes = []
        self.targets = []
        self.class_to_idx = {}
        self.modalities = []
        self.if_classification = if_classification
        self._write_attributes()
    
    def _write_attributes(self):
        for i, path in enumerate(self.paths):
            all_cls = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
            for cls in all_cls:
                imgs = [os.path.join(root, img) for root, _, imgs in os.walk(os.path.join(path, cls)) for img in imgs]
                if self.sample_per_class:
                    idx_L = random.choices(range(len(imgs)), k=self.sample_per_class)
                    imgs = [imgs[idx] for idx in idx_L]
                self.images.extend(imgs)
                self.classes.extend([cls]*len(imgs))
                self.modalities.extend([i]*len(imgs))
        # get the class to idx mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(np.unique(self.classes).tolist())}
        self.targets = [self.class_to_idx[x] for x in self.classes]
        assert len(self.images) == len(self.targets) == len(self.modalities)
        self.class_count_dict = {k: self.classes.count(k) for k in np.unique(self.classes)}

    def __getitem__(self, idx):
        if "tif" in self.images[idx % len(self.images)]:
            img = tifffile.imread(self.images[idx % len(self.images)])
        else:
            img = Image.open(self.images[idx % len(self.images)]).convert("RGB")
        img = np.array(img)
        if len(img.shape) != 3:
            img = np.stack([img, img, img], axis=-1)
        # if self.if_classification:
        #     to_tensor = v2.Compose([v2.ToImage(), 
        #                             v2.ToDtype(torch.float32),
        #                             v2.Resize((256,256)),])
        # else:
        to_tensor = v2.Compose([v2.ToImage(), 
                                v2.ToDtype(torch.float32),])
        img = to_tensor(img)# / 255 
        # img = img / img.max()

        img = (img - img.min()) / max(img.max()-img.min(), 1e-5)
        img = torch.clamp(img, max=1, min=0) #ensure no float overflow
        modality = self.modalities[idx % len(self.images)]
        label = self.targets[idx % len(self.images)]
        cls = self.classes[idx % len(self.images)]
        if self.transform:
            img = self.transform(img)
        # return the image as x and the class int label, modality, image path, and img cls as y
        return img, (label, modality, self.images[idx % len(self.images)], cls)
    
    def __len__(self):
        return len(self.images)
    
class Gfp_Dataset(Dataset):
    def __init__(self, path, transform=None):
        super(Gfp_Dataset, self).__init__()
        self.path = path
        self.images = [x for x in os.listdir(path) if x.endswith(".png")]
        self.transform = transform
    
    def __getitem__(self, idx):
        '''
        Image contains 2 images, left is GFP, right is PCM
        '''
        img = Image.open(os.path.join(self.path, self.images[idx])).convert("RGB")
        gfp = img.crop((0, 0, img.size[0]//2, img.size[1]))
        pcm = img.crop((img.size[0]//2, 0, img.size[0], img.size[1]))
        gfp = np.array(gfp, dtype=np.uint8)
        # compute gfp level, by identifying the percentage of green pixels
        green_channel = gfp[:, :, 1]
        threshold = 50  # based on image contrast
        green_pixels = green_channel > threshold
        percentage_green_pixels = np.sum(green_pixels) / (green_channel.shape[0] * green_channel.shape[1]) # range [0,1] so we can use softmax
        # process pcm image
        pcm = np.array(pcm, dtype=np.uint8)
        if len(pcm.shape) != 3:
            pcm = np.stack([pcm, pcm, pcm], axis=-1)
        # transformation
        to_tensor = v2.Compose([v2.ToImage(), 
                                v2.ToDtype(torch.float32),
                                v2.Resize((256,256)),])
        pcm = to_tensor(pcm) / 255 # rescale to [0,1]
        # pcm = to_tensor(pcm)
        # pcm = (pcm - pcm.min()) / max(pcm.max()-pcm.min(), 1e-5)
        pcm = torch.clamp(pcm, max=1.0, min=0.0) #ensure no float overflow
        if self.transform:
            pcm = self.transform(pcm)
        return pcm, (torch.tensor([percentage_green_pixels]).float(), os.path.join(self.path, self.images[idx]), "GFP", percentage_green_pixels)

    def __len__(self):
        return len(self.images)

class Vaibility_Dataset(Dataset):
    def __init__(self, path, transform=None, return_gfp=False):
        super(Vaibility_Dataset, self).__init__()
        self.return_gfp = return_gfp
        self.path = [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
        self.images = []
        self.treatment = []
        for each in self.path:
            pcm_image_path = [os.path.join(each, x) for x in os.listdir(each) if x.endswith("0.jpg")]
            self.images.extend(pcm_image_path)
            self.treatment.extend(os.path.basename(each).split("_")[2]*len(pcm_image_path))
        self.transform = transform
    
    def __getitem__(self, idx):
        '''
        Image contains 2 images, left is GFP, right is PCM
        '''
        pcm = Image.open(self.images[idx % len(self.images)]).convert("RGB")
        if self.return_gfp:
            colour = Image.open(self.images[idx % len(self.images)].replace("0.jpg", "1.jpg")).convert("RGB")
            colour = np.array(colour, dtype=np.uint8)
        else:
            green = Image.open(self.images[idx % len(self.images)].replace("0.jpg", "1.jpg")).convert("RGB")
            green = np.array(green, dtype=np.uint8)
            red = Image.open(self.images[idx % len(self.images)].replace("0.jpg", "2.jpg")).convert("RGB")
            red = np.array(red, dtype=np.uint8)
        
        # compute colour level, by identifying the percentage of green pixels
        threshold = 50
        if self.return_gfp:
            colour_channel = colour[:, :, 1]
            # based on image contrast
            colour_pixels = colour_channel > threshold
            percentage_colour_pixels = np.sum(colour_pixels) / (colour_channel.shape[0] * colour_channel.shape[1]) # range [0,1] 
        else:
            green = green[:, :, 1]
            red = red[:, :, 0]
            colour_green = green > threshold
            colour_red = red > threshold
            if ("Set_2" in self.images[idx % len(self.images)]) or ("Set_4" in self.images[idx % len(self.images)]):
                percentage_colour_pixels = 1 - (np.sum(colour_red) / (np.sum(colour_green) + np.sum(colour_red))) # range [0,1] 
            else:
                percentage_colour_pixels = 1 - (np.sum(colour_red) / (np.sum(colour_green)))
        # process pcm image
        pcm = np.array(pcm, dtype=np.uint8)
        if len(pcm.shape) != 3:
            pcm = np.stack([pcm, pcm, pcm], axis=-1)
        # transformation
        to_tensor = v2.Compose([v2.ToImage(), 
                                v2.ToDtype(torch.float32),
                                v2.Resize((256,256)),])
        pcm = to_tensor(pcm) / 255 # rescale to [0,1]
        # pcm = to_tensor(pcm)
        # pcm = (pcm - pcm.min()) / max(pcm.max()-pcm.min(), 1e-5)
        pcm = torch.clamp(pcm, max=1.0, min=0.0) #ensure no float overflow
        if self.transform:
            pcm = self.transform(pcm)
        return pcm, (torch.tensor([percentage_colour_pixels]).float(), self.images[idx % len(self.images)], 
                     "green" if self.return_gfp else "precent_alive",
                     percentage_colour_pixels)

    def __len__(self):
        return len(self.images)

class Wang_Segmentation_Dataset(Dataset):
    def __init__(self, path, transform=None):
        super(Wang_Segmentation_Dataset, self).__init__()
        self.path = path
        self.labels = []
        self.images = []
        labels = [x for x in os.listdir(os.path.join(path, "labels"))]
        for each in os.listdir(os.path.join(path, "images")):
            if f"{each.split('.')[0]}_label.tiff" in labels:
                self.images.append(os.path.join(path, "images", each))
                self.labels.append(os.path.join(path, "labels", f"{each.split('.')[0]}_label.tiff"))
        self.transform = transform
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, self.images[idx % len(self.images)])).convert("RGB")
        img = np.array(img, dtype=np.uint8)
        mask = Image.open(os.path.join(self.path, self.labels[idx % len(self.images)])).convert("L")
        mask = np.array(mask, dtype=np.uint8)
        # binary mask
        mask[mask > 0] = 1
        mask = np.stack([mask, mask, mask], axis=-1)
        if len(img.shape) != 3:
            img = np.stack([img, img, img], axis=-1)
        to_tensor_img = v2.Compose([v2.ToImage(), 
                                v2.ToDtype(torch.float32),
                                v2.Resize((256,256)),]) 
        to_tensor_mask = v2.Compose([v2.ToImage(),
                                v2.Resize((256,256)),])
        img = to_tensor_img(img) / 255 # rescale to [0,1]
        mask = to_tensor_mask(mask)
        # img = (img - img.min()) / max(img.max()-img.min(), 1e-5)
        img = torch.clamp(img, max=1, min=0) #ensure no float overflow
        mask = torch.clamp(mask, max=1, min=0)
        if self.transform:
            img = self.transform(img)
        return img, mask

    def __len__(self):
        return len(self.images)