'''
Generate pink filter images from the original images in a given directory.
'''

import os
import torch
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
from torch import nn
from utils.dataset import CustomImageDataset as data
import argparse
from tqdm import tqdm
import pytorch_lightning as pl
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

import torch
import torch.nn.functional as F
import numpy as np
import cv2

class MultiLayerGeneratorGradCAM:
    def __init__(self, model, target_layers_dict):
        """
        model: Your SPAGHETTI generator.
        target_layers_dict: A dictionary mapping your chosen names to the actual PyTorch modules.
                            e.g., {'block_1': model.res1, 'block_5': model.res5}
        """
        self.model = model
        self.target_layers_dict = target_layers_dict
        self.feature_maps = {}
        self.gradients = {}
        
        # Register hooks for every layer passed in the dictionary
        for name, layer in self.target_layers_dict.items():
            layer.register_forward_hook(self._save_feature_maps(name))
            layer.register_full_backward_hook(self._save_gradients(name))
            
    def _save_feature_maps(self, name):
        # Closure to capture the layer name
        def hook(module, input, output):
            self.feature_maps[name] = output
        return hook
        
    def _save_gradients(self, name):
        # Closure to capture the layer name
        def hook(module, grad_in, grad_out):
            self.gradients[name] = grad_out[0]
        return hook
        
    def generate_heatmaps(self, input_pcm_image, target_channel=None, roi_mask=None):
        self.model.eval()
        self.model.zero_grad()
        
        # 1. Forward Pass
        input_pcm_image.requires_grad_(True)
        generated_H_E = self.model(input_pcm_image)
        
        # 2. Define the Scalar Target (e.g., Blue/Hematoxylin channel)
        if roi_mask is not None:
            loss = (generated_H_E * roi_mask).sum()
        elif target_channel is not None:
            loss = generated_H_E[:, target_channel, :, :].sum()
        else:
            loss = generated_H_E.sum()
            
        # 3. Backward Pass (Populates self.gradients for all hooked layers)
        loss.backward()
        
        heatmaps = {}
        
        # 4. Compute Grad-CAM for each hooked layer
        for name in self.target_layers_dict.keys():
            gradients = self.gradients[name]
            features = self.feature_maps[name]
            
            # Global Average Pooling of gradients
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
            
            # Multiply features by weights
            cam = torch.sum(weights * features, dim=1, keepdim=True)
            
            # ReLU (Only keep features with positive influence)
            cam = F.relu(cam)
            
            # Upsample to match original image
            cam = F.interpolate(cam, size=input_pcm_image.shape[2:], mode='bilinear', align_corners=False)
            
            # Normalize
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            
            heatmaps[name] = cam.squeeze().detach().cpu().numpy()
            
        return heatmaps, generated_H_E.detach()

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # padding, keep the image size constant after next conv2d
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

class SpaghettiGenerator(nn.Module):
    def __init__(self, in_channels, num_residual_blocks=9):
        super(SpaghettiGenerator, self).__init__()

        self.normalization = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # Inital Convolution  3*256*256 -> 64*256*256
        out_channels=64
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(in_channels), # padding, keep the image size constant after next conv2d
            nn.Conv2d(in_channels, out_channels, 2*in_channels+1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        channels = out_channels
        
        # Downsampling   64*256*256 -> 128*128*128 -> 256*64*64
        self.down = []
        for _ in range(2):
            out_channels = channels * 2
            self.down += [
                nn.Conv2d(channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            channels = out_channels
        self.down = nn.Sequential(*self.down)
        
        # Transformation (ResNet)  256*64*64
        self.trans = [ResidualBlock(channels) for _ in range(num_residual_blocks)]
        self.trans = nn.Sequential(*self.trans)
        
        # Upsampling  256*64*64 -> 128*128*128 -> 64*256*256
        self.up = []
        for _ in range(2):
            out_channels = channels // 2
            self.up += [
                nn.Upsample(scale_factor=2), # bilinear interpolation
                nn.Conv2d(channels, out_channels, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            channels = out_channels
        self.up = nn.Sequential(*self.up)
        
        # Out layer  64*256*256 -> 3*256*256
        self.out = nn.Sequential(
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(channels, in_channels, 2*in_channels+1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.normalization(x)
        x = self.conv(x)
        x = self.down(x)
        x = self.trans(x)
        x = self.up(x)
        x = self.out(x)
        # normalize to range [0,1]
        x = torch.clamp(x, min=-1, max=1)
        min_val = x.min()
        max_val = x.max()
        x = (x-min_val)/(max(max_val-min_val, 1e-5))
        x = torch.clamp(x, min=0, max=1) # ensure no overflow
        return x

def init_spaghetti(model_path: str) -> torch.nn.Module:
    '''
    Initialize the SPAGHETTI model for image translation
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = SpaghettiGenerator(3, 9)
    generator.to(device)
    ckpt = torch.load(model_path, map_location=device)["state_dict"]
    # get only G_AB weights
    ckpt = {k[5:]: v for k, v in ckpt.items() if ("G_AB" in k)}
    generator.load_state_dict(ckpt)
    return generator

def init_spaghetti_reverse(model_path: str) -> torch.nn.Module:
    '''
    Initialize the SPAGHETTI model for image translation in the reverse direction
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = SpaghettiGenerator(3, 9)
    generator.to(device)
    ckpt = torch.load(model_path, map_location=device)["state_dict"]
    # get only G_BA weights
    ckpt = {k[5:]: v for k, v in ckpt.items() if ("G_BA" in k)}
    generator.load_state_dict(ckpt)
    return generator

def pink_filter_images(input_dir, output_dir, do_crop, model_path):
    os.makedirs(output_dir, exist_ok=True)

    # seed
    torch.manual_seed(42)
    pl.seed_everything(42)
    # Load the images from the input directory
    dataset = data(input_dir, transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Load the pre-trained model
    spaghetti_model = init_spaghetti(model_path)
    spaghetti_model.eval()
    spaghetti_reconstructor = init_spaghetti_reverse(model_path)
    spaghetti_reconstructor.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spaghetti_model.to(device)
    spaghetti_reconstructor.to(device)

    layers_to_inspect = {
        'ResBlock_1': spaghetti_model.trans[0],
        'ResBlock_3': spaghetti_model.trans[2],
        'ResBlock_6': spaghetti_model.trans[5],
        'ResBlock_9': spaghetti_model.trans[8]
    }

    cam_extractor = MultiLayerGeneratorGradCAM(spaghetti_model, layers_to_inspect)

    # with torch.no_grad():
    # Apply the pink filter to each image and save the result in the output directory
    for i, (image, labels) in enumerate(tqdm(dataloader)):
        image = image.to(device)
        f_name = labels[2][0].split("/")[-1].split(".")[0]
        if do_crop:
            # crop = v2.RandomCrop((256,256))
            crop = v2.RandomCrop((64,64))
            image = crop(image)
        # back to original size
        resize = v2.Resize((256,256))
        image = resize(image)
        pink_purple_color = torch.tensor([1.0, 182/255.0, 193/255.0]).view(3, 1, 1).to(device)
        # Apply the pink filter
        pink_image = 0.9 * image + 0.1 * pink_purple_color
        pink_image = torch.clamp(pink_image, 0, 1)

        # Apply the spaghetti model to generate the spaghetti image
        # invert = v2.RandomInvert(p=1.0)
        # image = invert(image)
        spaghetti_image = spaghetti_model(image)
        recon = spaghetti_reconstructor(spaghetti_image)
        #! Generate heatmaps (Targeting channel 2: Hematoxylin/Blue) for grad cam
        # input_pcm should be a tensor [1, C, H, W]
        heatmaps_dict, generated_img = cam_extractor.generate_heatmaps(image, target_channel=2)
        fig, axes = plt.subplots(1, len(heatmaps_dict) + 2, figsize=(20, 5))

        # Original Phase Contrast
        pcm_img_np = image.squeeze().detach().cpu().numpy()
        pcm_img_np = np.transpose(pcm_img_np, (1, 2, 0)) # Convert to HWC for plotting
        axes[0].imshow(pcm_img_np, cmap='gray' if pcm_img_np.shape[-1] == 1 else None)
        axes[0].set_title("Input PCM")
        axes[0].axis('off')

        # Plot each heatmap overlaid on the original image
        for idx, (layer_name, heatmap) in enumerate(heatmaps_dict.items()):
            ax = axes[idx + 1]
            
            # Optional: Convert to JET color map for overlay
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            # Ensure PCM is RGB for blending
            if pcm_img_np.shape[-1] == 1: 
                pcm_rgb = cv2.cvtColor(np.uint8(255 * pcm_img_np), cv2.COLOR_GRAY2RGB)
            else:
                pcm_rgb = np.uint8(255 * pcm_img_np)
                
            overlay = cv2.addWeighted(pcm_rgb, 0.4, heatmap_color, 0.6, 0)
            
            ax.imshow(overlay)
            ax.set_title(layer_name)
            ax.axis('off')

            # add colorbar
            cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Grad-CAM Intensity', rotation=270, labelpad=15)

        # Generated H&E
        gen_img_np = generated_img.squeeze().cpu().numpy()
        gen_img_np = np.transpose(gen_img_np, (1, 2, 0))
        # Denormalize if your output is [-1, 1]
        gen_img_np = (gen_img_np * 0.5) + 0.5 
        axes[-1].imshow(gen_img_np)
        axes[-1].set_title("Generated H&E")
        axes[-1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"spaghetti_image_{f_name}_gradcam_{i}.png"))
        plt.close()

        # save the images
        output_path = os.path.join(output_dir, f"pink_image_{f_name}.png")
        v2.ToPILImage()(pink_image.squeeze(0)).save(output_path)
        original_output_path = os.path.join(output_dir, f"spaghetti_image_{f_name}_original.png")
        v2.ToPILImage()(image.squeeze(0)).save(original_output_path)
        spaghetti_output_path = os.path.join(output_dir, f"spaghetti_image_{f_name}.png")
        v2.ToPILImage()(spaghetti_image.squeeze(0)).save(spaghetti_output_path)
        recon_output_path = os.path.join(output_dir, f"spaghetti_image_{f_name}_recon.png")
        v2.ToPILImage()(recon.squeeze(0)).save(recon_output_path)

def main():
    parser = argparse.ArgumentParser(description='Generate pink filter images from the original images in a given directory.')
    parser.add_argument('--input_dir', type=str, nargs="+", help='Directory containing the original images.')
    parser.add_argument('--output_dir', type=str, help='Directory to save the pink filter images.')
    parser.add_argument('--crop', action='store_true', help='Crop the images before applying the pink filter.')
    parser.add_argument('--model_path', type=str, help='Path to the pre-trained Spaghetti Generator model.')
    args = parser.parse_args()
    pink_filter_images(args.input_dir, args.output_dir, args.crop, args.model_path)

if __name__ == "__main__":
    main()
