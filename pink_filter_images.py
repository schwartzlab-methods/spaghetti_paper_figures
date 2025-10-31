'''
Generate pink filter images from the original images in a given directory.
'''

import os
import torch
import torchvision.transforms.v2 as v2
from torch import nn
from utils.dataset import CustomImageDataset as datset
import argparse
from tqdm import tqdm

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

def pink_filter_images(input_dir, output_dir, do_crop, model_path):
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the images from the input directory
    dataset = datset(input_dir, transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Load the pre-trained model
    spaghetti_model = init_spaghetti(model_path)
    spaghetti_model.eval()

    with torch.no_grad():
        # Apply the pink filter to each image and save the result in the output directory
        for i, (image, _) in enumerate(tqdm(dataloader)):
            if do_crop:
                crop = v2.RandomCrop((256,256))
                image = crop(image)
            # back to original size
            resize = v2.Resize((256,256))
            image = resize(image)
            pink_purple_color = torch.tensor([1.0, 182/255.0, 193/255.0]).view(3, 1, 1)
            # Apply the pink filter
            pink_image = 0.9 * image + 0.1 * pink_purple_color
            pink_image = torch.clamp(pink_image, 0, 1)

            # Save the pink filter image
            output_path = os.path.join(output_dir, f"pink_image_{i}.png")
            v2.ToPILImage()(pink_image.squeeze(0)).save(output_path)

            # save the original image for comparison
            original_output_path = os.path.join(output_dir, f"original_image_{i}.png")
            v2.ToPILImage()(image.squeeze(0)).save(original_output_path)

            # Apply the spaghetti model to generate the spaghetti image
            invert = v2.RandomInvert(p=1.0)
            spaghetti_image = spaghetti_model(invert(pink_image))
            spaghetti_output_path = os.path.join(output_dir, f"spaghetti_image_{i}.png")
            v2.ToPILImage()(spaghetti_image.squeeze(0)).save(spaghetti_output_path)

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