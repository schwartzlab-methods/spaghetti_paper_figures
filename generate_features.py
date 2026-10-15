'''
Generate features for the dataset
'''
import torch
import random
from huggingface_hub import login
from utils.dataset import CustomImageDataset, Gfp_Dataset, Vaibility_Dataset
from utils.utils import find_checkpoint, prep_datasets, loader_data, map_labels, plot_masks, str2bool#, sample_imgs 
from utils.utils import image_transform as trans_cycleGAN
from feature_extractors import owkin_features, vit_features, resnet_features, h_optimus_0_features, uni_features
from torchvision.models import vit_l_16, resnet18
import pytorch_lightning as pl
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms.v2 as v2
import argparse
import os
import numpy as np
import cycleGAN as c
import timm
from sklearn.decomposition import PCA
import altair as alt
import pandas as pd
from tqdm import tqdm
## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

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

def main():
    # prepare the feature extractor
    if args.convert:
        cycleGAN_gen = init_cycleGAN(args.convert)
        process_cycleGAN = trans_cycleGAN(do_augmentation=False, do_cropping=False) #will do crop later if needed
        convert_to_HE = lambda x, device: cycleGAN_transform(cycleGAN_gen, device, process_cycleGAN, x)
    else:
        convert_to_HE = None
    if args.feature_extractor == 'phikon':
        feature_extractor = AutoModel.from_pretrained("/fs01/home/richarddong/.cache/huggingface/hub/phikon-v2")
        image_processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        feature = lambda x, device, train: owkin_features(feature_extractor, device, image_processor, x, 
                                                        convert=convert_to_HE, invert=False, do_filter=False, 
                                                        do_crop=False, train=train)
    elif args.feature_extractor == 'vit':
        feature_extractor = vit_l_16()
        feature = lambda x, device, train: vit_features(feature_extractor, device, x, 
                                                        convert=convert_to_HE, invert=False, do_filter=False,
                                                        do_crop=False, train=train)
    elif args.feature_extractor == 'resnet':
        feature_extractor = resnet18(weights='DEFAULT')
        feature = lambda x, device, train: resnet_features(feature_extractor, device, x, 
                                                        convert=convert_to_HE, invert=False, do_filter=False,
                                                        do_crop=False, train=train)
    elif args.feature_extractor == 'h-optimus':
        login(token=args.hugging_face_token)
        feature_extractor = timm.create_model(
                    "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
        feature = lambda x, device, train: h_optimus_0_features(feature_extractor, device, x, 
                                                                convert=convert_to_HE, invert=False, do_filter=False,
                                                                do_crop=False, train=train)
    elif args.feature_extractor == 'uni':
        login(token=args.hugging_face_token)
        feature_extractor = timm.create_model(
                    "hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        feature = lambda x, device, train: uni_features(feature_extractor, device, x,
                                                        convert=convert_to_HE, invert=False, do_filter=False,
                                                        do_crop=False, train=train)
    else:
        raise ValueError('Invalid feature extractor')
    # prepare the dataset
    if args.gfp_data:
        data = Gfp_Dataset(args.image_path[0], transform=None)
    elif args.gfp_data_russian:
        data = Vaibility_Dataset(args.image_path[0], transform=None, return_gfp=True)
    else:
        data = CustomImageDataset(args.image_path, transform=None)
        class_to_idx = data.class_to_idx
        print("Data prep finished. The class to idx is: ", class_to_idx)
    if args.data_indices:
        indices = []
        for each in args.data_indices:
            with open(each, "r") as f:
                idx = f.readlines()
                idx = [int(x) for x in idx]
            indices.append(idx)
        _, data = prep_datasets(data, indices)
    total_loader = loader_data(data, batch_size=1)
    print("Data loaded, size: ", len(total_loader))
    # get the features for each
    features_L = []
    labels_L = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for img, labels in tqdm(total_loader):
        img = img.to(device)
        # get the features
        extracted = feature(img, device, train=False)
        extracted = extracted.cpu().numpy()
        features_L.append(extracted)
        # get the labels
        label = labels[3][0].cpu().numpy()
        labels_L.append(label)
    features_np = np.concatenate(features_L, axis=0)
    labels_np = np.array(labels_L)
    if args.convert:
        convert_status = "spaghetti"
    else:
        convert_status = "no_spaghetti"
    np.save(os.path.join(args.save_dir, f"features_{args.feature_extractor}_{convert_status}_{args.name}.npy"), features_np)
    np.save(os.path.join(args.save_dir, f"labels_{args.feature_extractor}_{convert_status}_{args.name}.npy"), labels_np)
    # plot PCA of features
    pca = PCA(n_components=50)
    embedding_original = pca.fit_transform(features_np)
    var_ex_original = pca.explained_variance_ratio_

    # prep pandas for altair scatter
    df_original = pd.DataFrame({f"PC 1": embedding_original[:, 0],
                                f"PC 2": embedding_original[:, 1],
                                "Classes": labels_np})
    # plot with altair
    scatter1 = alt.Chart(df_original).mark_point().encode(
        x=alt.X(f"PC 1", title=f"PC 1 (Variance Explained = {var_ex_original[0] * 100:.2f}%)"),
        y=alt.Y(f"PC 2", title=f"PC 2 (Variance Explained = {var_ex_original[1] * 100:.2f}%)"),
        color="Classes",
    )
    scatter1.interactive().save(os.path.join(args.save_dir, f"feature_pca_{args.feature_extractor}_{convert_status}_{args.name}.html"))
    # quantify the features from PCA
    corr = np.corrcoef(embedding_original[:, 0], labels)[0, 1]
    print(f"Correlation coefficients: {corr}")
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='Name of the experiment')
    parser.add_argument('--hugging_face_token', type=str, default=None, 
                        help='Access token for Hugging Face Hub to use with pre-trained models (required for H-Optimus and Uni)')
    parser.add_argument('--feature_extractor', type=str, default='phikon', 
                        help='Feature extractor. Possible values: phikon, vit, h-optimus, or resnet')
    parser.add_argument('--data_indices', type=str, nargs='+', 
                        help='Path(s) to the txt files containing indicies for data split')
    parser.add_argument('--image_path', type=str, nargs='+', help='Path(s) to the image')
    parser.add_argument('--save_dir', type=str, help='Path to save the results')
    parser.add_argument('--gfp_data', action="store_true", help='Use the Japanese GFP data')
    parser.add_argument('--gfp_data_russian', action="store_true", help='Use the Russian GFP data')
    parser.add_argument('--convert', type=str, default=None, 
                        help='Convert the images to H&E using trained CycleGAN models. Put path to the model here')
    args = parser.parse_args()
    print("Program starting... Parameters:")
    print(args)
    os.makedirs(args.save_dir, exist_ok=True)
    # seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    pl.seed_everything(42, workers=True)
    # run!
    main()