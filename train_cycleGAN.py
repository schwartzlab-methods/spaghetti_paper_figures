import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import utils
from utils import dataset
from utils import utils
from torchvision.utils import save_image
from torchvision.transforms.v2 import Grayscale
import cycleGAN
from pytorch_lightning.loggers import CSVLogger
import os
import pytorch_lightning as pl
from PIL import ImageFile
import numpy as np
import argparse
import itertools
import random

class LitModel(pl.LightningModule):
    def __init__(self, batch_size, do_crop = False, weights = [1.0, 10.0, 5.0, 3.0]):
        super().__init__()
        self.automatic_optimization = False
        # model
        self.G_AB = cycleGAN.GeneratorResNet(3, 9)
        self.D_B = cycleGAN.Discriminator(3)
        self.G_BA = cycleGAN.GeneratorResNet(3, 9)
        self.D_A = cycleGAN.Discriminator(3)
        # loss
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_ssim = cycleGAN.SSIMLoss()
        self.weights = weights
        # others
        self.batch_size = batch_size
        self.do_crop = do_crop

    def calculate_loss_generator(self, res, x1, x2):
        # groud truth
        out_shape = [x1.size(0), 1, x1.size(2)//self.D_A.scale_factor, 
                     x1.size(3)//self.D_A.scale_factor]
        valid = torch.ones(out_shape).to(self.device)
        fake_x1, fake_x2, recov_x1, recov_x2, new_x1, new_x2 = res
        # convert to gray scale for ssim calculation
        gray = Grayscale(num_output_channels=3)
        x1_gray = gray(x1)
        x2_gray = gray(x2)
        fake_x1_gray = gray(fake_x1)
        fake_x2_gray = gray(fake_x2)
        new_x1_gray = gray(new_x1)
        new_x2_gray = gray(new_x2)
        loss_GAN = (self.criterion_GAN(self.D_A(fake_x1), valid) 
                    + self.criterion_GAN(self.D_B(fake_x2), valid))/2
        loss_cycle = (self.criterion_cycle(recov_x1, x1) 
                      + self.criterion_cycle(recov_x2, x2))/2
        loss_identity = (self.criterion_identity(new_x1, x1) 
                         + self.criterion_identity(new_x2, x2))/2
        loss_ssim_fake = (self.criterion_ssim(fake_x2_gray, x1_gray) 
                     + self.criterion_ssim(fake_x1_gray, x2_gray)) / 2
        loss_ssim_real = (self.criterion_ssim(new_x1_gray, x1_gray)
                        + self.criterion_ssim(new_x2_gray, x2_gray)) / 2
        loss_ssim = (loss_ssim_fake + loss_ssim_real) / 2
        total_loss = (self.weights[0] * loss_GAN + self.weights[1] * loss_cycle 
                      + self.weights[2] * loss_identity + self.weights[3] * loss_ssim)
        return total_loss

    def calculate_loss_discriminator(self, dis, x, x_fake):
        out_shape = [x.size(0), 1, x.size(2)//dis.scale_factor, 
                     x.size(3)//dis.scale_factor]
        valid = torch.ones(out_shape).to(self.device)
        fake = torch.zeros(out_shape).to(self.device)
        loss_real = self.criterion_GAN(dis(x), valid)
        loss_fake = self.criterion_GAN(dis(x_fake.detach()), fake)
        total_loss = (loss_real + loss_fake) /2
        return total_loss
    
    def training_step(self, batch, batch_idx):
        optimizer_G, optimizer_D_A, optimizer_D_B = self.optimizers()
        accumulated_grad_batches = (batch_idx+1) % self.batch_size == 0
        x1, x2 = batch
        # for discriminator
        fake_x1 = self.G_BA(x2)
        fake_x2 = self.G_AB(x1)
        
        # for identity
        new_x1 = self.G_BA(x1)
        new_x2 = self.G_AB(x2)

        # for reconstruction
        recov_x2 = self.G_AB(fake_x1)
        recov_x1 = self.G_BA(fake_x2)
        
        res = (fake_x1, fake_x2, recov_x1, recov_x2, new_x1, new_x2)
        
        # generator loss
        gen_loss = self.calculate_loss_generator(res, x1, x2)

        # discriminator A loss
        d_a_loss = self.calculate_loss_discriminator(self.D_A, x1, fake_x1)
        
        # discriminator B loss
        d_b_loss = self.calculate_loss_discriminator(self.D_B, x2, fake_x2)

        # optmize
        total_loss = gen_loss + d_a_loss + d_b_loss
        self.manual_backward(total_loss)
        
        if accumulated_grad_batches:
            optimizer_G.step()
            optimizer_G.zero_grad()
            optimizer_D_B.step()
            optimizer_D_B.zero_grad()
            optimizer_D_A.step()
            optimizer_D_A.zero_grad()

        metrics = {'train_gen_loss': gen_loss, 'train_D_A_loss': d_a_loss, 'train_D_B_loss': d_b_loss}
        self.log_dict(metrics,prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x1, x2 = batch
        fake_x1 = self.G_BA(x2)
        fake_x2 = self.G_AB(x1)
        
        # for identity
        new_x1 = self.G_BA(x1)
        new_x2 = self.G_AB(x2)

        # for reconstruction
        recov_x2 = self.G_AB(fake_x1)
        recov_x1 = self.G_BA(fake_x2)
        
        res = (fake_x1, fake_x2, recov_x1, recov_x2, new_x1, new_x2)

        loss = self.calculate_loss_generator(res, x1, x2)
        self.log_dict({'val_gen_loss': loss},sync_dist=True,prog_bar=True)
        # save image for visualization
        visual = torch.cat((x1, fake_x2, x2, fake_x1), 0)
        if batch_idx % 1000 == 0:
            try:
                save_image(visual, os.path.join(args.save_dir, "visual", f"visual_{str(self.global_rank)}_{batch_idx}_epoch_{self.current_epoch}.png"), 
                        nrow=4, normalize=True, value_range=(-1, 1))
            except FileExistsError:
                pass 

    def configure_optimizers(self):
        optimizer_G = torch.optim.AdamW(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), 
            lr=args.lr, weight_decay=1e-2)
        optimizer_D_A = torch.optim.AdamW(
            self.D_A.parameters(), 
            lr=args.lr, weight_decay=1e-2)
        optimizer_D_B = torch.optim.AdamW(
            self.D_B.parameters(), 
            lr=args.lr, weight_decay=1e-2)
        return [optimizer_G, optimizer_D_A, optimizer_D_B], []

def main(num_nodes, ngpus_per_node):
    # create data
    if not args.gfp_data:
        data = dataset.CycleGANDataset(args.path_1, args.path_2, num_sample=args.num_samples, 
                                    transform=utils.image_transform, do_crop_1=args.crop, keep_RGB_img=args.RGB)
        # split dataset and save indicies for reproducibility
        train_dataset, val_dataset = torch.utils.data.random_split(data, [0.5, 0.5])
    else:
        data = dataset.CycleGANDatasetForGFP(args.path_1, args.path_2, num_sample=args.num_samples, 
                                    transform=utils.image_transform, do_crop_1=args.crop, keep_RGB_img=args.RGB)
    
        # split dataset and save indicies for reproducibility
        train_dataset, val_dataset = torch.utils.data.random_split(data, [0.8, 0.2])
    with open(os.path.join(args.save_dir,"train_indicies.txt"), "w") as f:
        for idx in train_dataset.indices:
            f.write("%s\n" % idx)
    with open(os.path.join(args.save_dir,"val_indicies.txt"), "w") as f:
        for idx in val_dataset.indices:
            f.write("%s\n" % idx)
    print("Data splited to train and test, indicies saved to .txt")

    # create dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # create model
    lit_model = LitModel(batch_size=args.batch_size, do_crop=args.crop, weights=args.weights)
    # train model
    logger = CSVLogger(args.save_dir, name=args.name)
    trainer = pl.Trainer(max_epochs=args.epochs, devices=ngpus_per_node, num_nodes=num_nodes,
                         use_distributed_sampler=True, enable_progress_bar=True,
                         strategy="ddp",
                         default_root_dir=args.save_dir, logger=logger)
    print("Trainer initialized with ", ngpus_per_node, "GPU(s) per node on ", num_nodes, "node(s)")
    print("Training Starting...")
    ckpt = utils.find_checkpoint(args.save_dir)
    if ckpt:
        print("Checkpoint found. Resuming from ", ckpt)
    else:
        print("Starting from epoch 0")
    trainer.fit(lit_model, train_loader, val_loader, None, ckpt)
    print("Training ended.")

if __name__ == '__main__':
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='Name of the experiment')
    parser.add_argument('--path_1', type=str, help='Path to the first (target) domain')
    parser.add_argument('--path_2', type=str, nargs="+", help='Path(s) to the second (source) domain(s)')
    parser.add_argument('--num_samples', type=int, default=None, help='Num of samples per path for second (source) domain')
    parser.add_argument('--RGB', action="store_true", help='use RGB of images instead of defaultgray scale')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weights', type=float, nargs="+", default=[1.0, 10.0, 5.0, 3.0], 
                        help="Weights for different loss in the order of GAN, cycle, identity, and SSIM.")
    parser.add_argument('--crop', action="store_true", help='Crop the image to be a quartre of the size on domain 1')
    parser.add_argument('--gfp_data', action="store_true", help='train with gfp data')
    args = parser.parse_args()
    print("parameters: ")
    print(args)

    assert len(args.weights) == 4, "Length of weights must be 4"
    # params
    ngpus_per_node = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    print(f"Local rank: {local_rank}")
    num_nodes = int(os.environ.get("SLURM_NNODES"))
    rank = num_nodes*ngpus_per_node + local_rank
    print(f"Total rank: {rank}")
    current_device = local_rank
    torch.cuda.set_device(current_device)
    # print all args for debugging
    print("Arguments for the program: ")
    print(args)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "visual"), exist_ok=True)

    main(num_nodes, ngpus_per_node)
