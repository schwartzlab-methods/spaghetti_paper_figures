'''
Infer the GFP amount from the image
'''
import torch
import random
from huggingface_hub import login
import downstream_models as dm
from utils.dataset import Gfp_Dataset, Vaibility_Dataset
from utils.utils import find_checkpoint, prep_datasets, loader_data, str2bool
from utils.utils import image_transform as trans_cycleGAN
from feature_extractors import owkin_features, vit_features, resnet_features, h_optimus_0_features, uni_features
from torchvision.models import vit_l_16, resnet18
import pytorch_lightning as pl
from transformers import AutoImageProcessor, AutoModel
import argparse
import os
import numpy as np
import pandas as pd
import cycleGAN as c
import timm

def init_cycleGAN(model_path: str) -> torch.nn.Module:
    '''
    Initialize the CycleGAN model
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = c.GeneratorResNet(3, 9)
    generator.to(device)
    ckpt = torch.load(model_path, map_location=device)["state_dict"]
    # get only G_AB weights
    ckpt = {k[5:]: v for k, v in ckpt.items() if ("G_AB" in k)}
    generator.load_state_dict(ckpt)
    return generator

def cycleGAN_transform(cycleGAN_model, device, transform, x):
    model = cycleGAN_model.to(device)
    model.eval()
    with torch.no_grad():
        x = transform(x)
        out = model(x.to(device))
    # normalize to range [0,1]
    out = torch.clamp(out, min=-1, max=1)
    min_val = out.min()
    max_val = out.max()
    out = (out-min_val)/(max(max_val-min_val, 1e-5))
    out = torch.clamp(out, min=0, max=1) # ensure no overflow
    return out

def train_task(model, feature_extractor, train_loader, val_loader, num_epochs, task):
    ngpus_per_node = torch.cuda.device_count()
    num_nodes = int(os.environ.get("SLURM_NNODES"))
    pl_model = dm.build_model(model, feature_extractor, task=task)
    logger = pl.loggers.CSVLogger(args.save_dir, name=args.name)
    trainer = pl.Trainer(max_epochs=num_epochs, devices=ngpus_per_node, num_nodes=num_nodes,
                         use_distributed_sampler=True, enable_progress_bar=True,strategy="ddp",
                         default_root_dir=args.save_dir, logger=logger)
    print("Training starting")
    ckpt = find_checkpoint(args.save_dir)
    trainer.fit(pl_model, train_loader, val_loader, None, ckpt)
    print("Training finished")
    return trainer, pl_model

def main():
    if args.russian_dataset:
        if args.do_gfp:
            data = Vaibility_Dataset(args.image_path, transform=None, return_gfp=True)
        else:
            data = Vaibility_Dataset(args.image_path, transform=None, return_gfp=False)
    else:
        data = Gfp_Dataset(args.image_path, transform=None)
    if args.data_indices:
        indices = []
        for each in args.data_indices:
            with open(each, "r") as f:
                idx = f.readlines()
                idx = [int(x) for x in idx]
            indices.append(idx)
        _, data = prep_datasets(data, indices)
    if args.method == "lr-cv" or args.method == "mlp-cv":
        total_loader = loader_data(data, batch_size=1)
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(data, [args.training_percentage*0.01,1-args.training_percentage*0.01])
        val_loader = loader_data(val_dataset, batch_size=1)
        train_loader = loader_data(train_dataset, batch_size=16, shuffle=True)
    # prepare the feature extractor
    if args.convert:
        cycleGAN_gen = init_cycleGAN(args.convert)
        process_cycleGAN = trans_cycleGAN(do_augmentation=False, do_cropping=False)
        convert_to_HE = lambda x, device: cycleGAN_transform(cycleGAN_gen, device, process_cycleGAN, x)
    else:
        convert_to_HE = None
    if args.feature_extractor == 'phikon':
        feature_extractor = AutoModel.from_pretrained("owkin/phikon-v2")
        image_processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        feature = lambda x, device, train: owkin_features(feature_extractor, device, image_processor, x, 
                                                        convert=convert_to_HE, invert=args.do_invert, do_filter=False, 
                                                        do_crop=False, train=train)
    elif args.feature_extractor == 'vit':
        feature_extractor = vit_l_16()
        feature = lambda x, device, train: vit_features(feature_extractor, device, x, 
                                                        convert=convert_to_HE, invert=args.do_invert, do_filter=False,
                                                        do_crop=False, train=train)
    elif args.feature_extractor == 'resnet':
        feature_extractor = resnet18(weights='DEFAULT')
        feature = lambda x, device, train: resnet_features(feature_extractor, device, x, 
                                                        convert=convert_to_HE, invert=args.do_invert, do_filter=False,
                                                        do_crop=False, train=train)
    elif args.feature_extractor == 'h-optimus':
        login(token=args.hugging_face_token)
        feature_extractor = timm.create_model(
                    "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
        feature = lambda x, device, train: h_optimus_0_features(feature_extractor, device, x, 
                                                                convert=convert_to_HE, invert=args.do_invert, do_filter=False,
                                                                do_crop=False, train=train)
    elif args.feature_extractor == 'uni':
        login(token=args.hugging_face_token)
        feature_extractor = timm.create_model(
                    "hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        feature = lambda x, device, train: uni_features(feature_extractor, device, x,
                                                        convert=convert_to_HE, invert=args.do_invert, do_filter=False,
                                                        do_crop=False, train=train)
    else:
        raise ValueError('Invalid feature extractor')
    # #* regression with deep learning
    if args.method == 'nn':
        input_size = (512 if args.feature_extractor == "resnet" 
                        else 1536 if args.feature_extractor == "h-optimus"  
                        else 1024)
        task_model = dm.Regression(input_size)
        trainer, pl_model = train_task(task_model, feature, train_loader, val_loader, 
                                        args.num_epochs, task="regression")
        print("Inference starting")
        trainer.predict(model=pl_model, dataloaders=val_loader, return_predictions=False)
        predictions = pl_model.pred
        true = pl_model.true
        loss = pl_model.loss
        # save to files
        np.save(os.path.join(args.save_dir, f"predictions_{args.feature_extractor}_{args.name}.npy"), predictions)
        np.save(os.path.join(args.save_dir, "true.npy"), true)
        np.save(os.path.join(args.save_dir, f"loss_{args.feature_extractor}_{args.name}.npy"), loss)
    elif args.method == 'mlp-cv':
        input_size = (512 if args.feature_extractor == "resnet" 
                        else 1536 if args.feature_extractor == "h-optimus"  
                        else 1024)
        r_value, pred, tru = dm.mlp_with_cv(total_loader, feature,
                                                       torch.device('cuda' if torch.cuda.is_available() else 'cpu'), input_size)
        data = pd.DataFrame({"pred": pred, "true": tru})
        data.to_csv(os.path.join(args.save_dir, f"predictions_{args.feature_extractor}_{args.name}.csv"))
        np.save(os.path.join(args.save_dir, f"rvalues_{args.feature_extractor}_{args.name}.npy"), np.array(r_value))
    elif args.method == 'lr-cv':
        r_value, pred, tru = dm.linear_regression_with_cv(total_loader, feature, 
                                                          torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        data = pd.DataFrame({"pred": pred, "true": tru})
        data.to_csv(os.path.join(args.save_dir, f"predictions_{args.feature_extractor}_{args.name}.csv"))
        # np.save(os.path.join(args.save_dir, f"predictions_{args.feature_extractor}_{args.name}.npy"), np.array(pred))
        # np.save(os.path.join(args.save_dir, f"true_{args.feature_extractor}_{args.name}.npy"), np.array(tru))
        np.save(os.path.join(args.save_dir, f"rvalues_{args.feature_extractor}_{args.name}.npy"), np.array(r_value))
    else:
        #* regression with linear regression from sklearn
        pred, tru = dm.linear_regression(train_loader, val_loader, feature, 
                                        torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # save to files
        np.save(os.path.join(args.save_dir, f"predictions_{args.feature_extractor}_{args.name}.npy"), pred)
        np.save(os.path.join(args.save_dir, "true.npy"), tru)
    print("Task completed")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='Name of the experiment')
    parser.add_argument('--data_indices', type=str, nargs='+', 
                        help='Path(s) to the txt files containing indicies for data split')
    parser.add_argument("--do_invert", action="store_true", help="If invert the input")
    parser.add_argument('--training_percentage', type=int, default=10, help='amount of data for training')
    parser.add_argument('--hugging_face_token', type=str, default=None, 
                        help='Access token for Hugging Face Hub to use with pre-trained models (required for H-Optimus and Uni)')
    parser.add_argument('--feature_extractor', type=str, default='phikon', 
                        help='Feature extractor. Possible values: phikon, vit, h-optimus, uni, or resnet')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Path to the model for feature extraction. If none, use ImageNet weights')
    parser.add_argument('--image_path', type=str, help='Path to the image')
    parser.add_argument('--save_dir', type=str, help='Path to save the results')
    parser.add_argument('--convert', type=str, default=None, 
                        help='Convert the images to H&E using trained CycleGAN models. Put path to the model here')
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs, only for logistic regression")
    parser.add_argument('--method', type=str, default='nn',
                        help='Method to use for inference. Possible values: nn, lr, lr-cv')
    parser.add_argument("--russian_dataset", action="store_true", help="Use the russian dataset")
    parser.add_argument('--do_gfp', action="store_true", help="If use GFP")
    args = parser.parse_args()
    print("Program starting... Parameters:")
    print(args)
    # run!
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    main()
