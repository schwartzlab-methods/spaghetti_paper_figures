'''
Downstream task validation for the feature extracted from ViT
Pssible tasks are:
1. classification (cell type or growth factor)
2. segmentation with CellPose models
3. regression for cell viability
'''
import torch
import random
from huggingface_hub import login
import downstream_models as dm
from utils.dataset import CustomImageDataset, Wang_Segmentation_Dataset
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
from pycocotools.coco import COCO
import timm
from plot_curves import plot_confusion_matrix, plot_ROC_curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

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

def compute_auc(pred_L, true_L, cls_to_idx):
    # Compute ROC curve and ROC area for each class
    auc_L = []
    for fold in range(len(pred_L)):
        true_bin = label_binarize(true_L[fold], classes=list(cls_to_idx.values()))
        pred = np.array(pred_L[fold])
        # Compute micro-average ROC curve and ROC area (aggregate over all classes)
        fpr_array, tpr_array, _ = roc_curve(true_bin.ravel(), pred.ravel())
        auc_L.append(auc(fpr_array, tpr_array))
    return auc_L

def main():
    if args.crop and (args.task != "segmentation"):
        transform_ops = [v2.RandomCrop(size=args.crop_size), v2.Resize((256,256))]
        transform = v2.Compose(transform_ops)
    else:
        transform = None
    if args.wang_data:
        data = Wang_Segmentation_Dataset(args.image_path[0], transform=transform)
    else:
        data = CustomImageDataset(args.image_path, transform=transform, if_classification=args.task != "segmentation")
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
    train_dataset, val_dataset = prep_datasets(data)
    total_loader = loader_data(data, batch_size=1)
    val_loader = loader_data(val_dataset, batch_size=1)
    train_loader = loader_data(train_dataset, batch_size=16, shuffle=True)
    # prepare the feature extractor
    if args.convert:
        cycleGAN_gen = init_cycleGAN(args.convert)
        process_cycleGAN = trans_cycleGAN(do_augmentation=False, do_cropping=False) #will do crop later if needed
        convert_to_HE = lambda x, device: cycleGAN_transform(cycleGAN_gen, device, process_cycleGAN, x)
    else:
        convert_to_HE = None
    if args.task != 'segmentation':
        if args.feature_extractor == 'phikon':
            feature_extractor = AutoModel.from_pretrained("owkin/phikon-v2")
            image_processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
            feature = lambda x, device, train: owkin_features(feature_extractor, device, image_processor, x, 
                                                            convert=convert_to_HE, invert=args.invert, do_filter=args.do_colour_filter, 
                                                            do_crop=False, train=train)
        elif args.feature_extractor == 'vit':
            feature_extractor = vit_l_16()
            feature = lambda x, device, train: vit_features(feature_extractor, device, x, 
                                                            convert=convert_to_HE, invert=args.invert, do_filter=args.do_colour_filter,
                                                            do_crop=False, train=train)
        elif args.feature_extractor == 'resnet':
            feature_extractor = resnet18(weights='DEFAULT')
            feature = lambda x, device, train: resnet_features(feature_extractor, device, x, 
                                                            convert=convert_to_HE, invert=args.invert, do_filter=args.do_colour_filter,
                                                            do_crop=False, train=train)
        elif args.feature_extractor == 'h-optimus':
            login(token=args.hugging_face_token)
            feature_extractor = timm.create_model(
                        "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
            feature = lambda x, device, train: h_optimus_0_features(feature_extractor, device, x, 
                                                                    convert=convert_to_HE, invert=args.invert, do_filter=args.do_colour_filter,
                                                                    do_crop=False, train=train)
        elif args.feature_extractor == 'uni':
            login(token=args.hugging_face_token)
            feature_extractor = timm.create_model(
                        "hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
            feature = lambda x, device, train: uni_features(feature_extractor, device, x,
                                                            convert=convert_to_HE, invert=args.invert, do_filter=args.do_colour_filter,
                                                            do_crop=False, train=train)
        else:
            raise ValueError('Invalid feature extractor')
    if args.task == 'logistic_regression':
        input_size = (512 if args.feature_extractor == "resnet" 
                      else 1536 if args.feature_extractor == "h-optimus"  
                      else 1024)
        task_model = dm.Classification(len(class_to_idx), input_size)
        trainer, pl_model = train_task(task_model, feature, train_loader, val_loader, 
                                       args.num_epochs, args.task)
        print("Inference starting")
        trainer.predict(model=pl_model, dataloaders=val_loader, return_predictions=False)
        predictions = pl_model.pred
        true = pl_model.true
        predictions_prob = pl_model.prob
        predict_str = map_labels(predictions, class_to_idx)
        true_str = map_labels(true, class_to_idx)
        plot_confusion_matrix(predict_str, true_str, args.save_dir, args.name)
        plot_ROC_curve(pred_L=[predictions_prob], true_L=[true], cls_to_idx=class_to_idx,
                       save=args.save_dir, name=args.name)
    if args.task == "random_forest":
        # only runs on 1 GPU
        save_dir = os.path.join(args.save_dir, args.name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.do_cross_val:
            save_path = os.path.join(save_dir, f"rf_features_labels_{args.name}.npz")
            pred_prob_L, true_L = dm.random_forest_classifier(total_loader, feature, device, save_path=save_path, load=save_path if args.load else None)
            plot_ROC_curve(pred_L=pred_prob_L, true_L=true_L, cls_to_idx=class_to_idx,
                            save=save_dir, name=args.name)
        elif args.test_for_noise:
            final_dict = dm.random_forest_with_increasing_noise(total_loader, feature, device)
            for k in final_dict.keys():
                pred_prob_L, true_L, data_images = final_dict[k]
                auc_L = compute_auc(pred_prob_L, true_L, class_to_idx)
                with open(os.path.join(save_dir, f"auc_{args.name}_{k}_noise.txt"), 'w') as f:
                    for item in auc_L:
                        f.write("%s\n" % item)
                # save images
                plot_ROC_curve(pred_L=pred_prob_L, true_L=true_L, cls_to_idx=class_to_idx,
                            save=save_dir, name=f"{args.name}_{k}_noise")
                # save the first 5 images
                fig, ax = plt.subplots(1, 5, figsize=(20, 8))
                for i in range(min(5, len(data_images))):
                    ax[i].imshow(data_images[i].transpose(1,2,0))
                    ax[i].axis('off')
                plt.savefig(os.path.join(save_dir, f"images_{k}_noise.png"))
                plt.close()
        else: 
            pred_prob_L, true_L, percent_train = dm.random_forest_with_increasing_training_size(total_loader, feature, device)
            auc_L = compute_auc(pred_prob_L, true_L, class_to_idx)
            with open(os.path.join(save_dir, f"auc_{args.name}.txt"), 'w') as f:
                for item in auc_L:
                    f.write("%s\n" % item)
            with open(os.path.join(save_dir, f"percent_train_{args.name}.txt"), 'w') as f:
                for item in percent_train:
                    f.write("%s\n" % item)
    elif args.task == "segmentation":
        assert len(args.segmentation_model) == len(args.do_convert), "Number of models and conversions mismatch"
        if args.coco:
            coco_L = [COCO(coco) for coco in args.coco]
            dice_L, seg_result_nonbinary, seg_result_binary, img_L, extra_pixels_L = dm.segmentation(total_loader, convert_to_HE, args.segmentation_model, 
                                                                            do_convert=args.do_convert, do_invert=args.do_invert, do_crop=args.crop, coco=coco_L,
                                                                            utom=args.utom)
        elif args.masks:
            masks_L = []
            for mask_dir in args.masks:
                masks = [os.path.join(mask_dir, x) for x in os.listdir(mask_dir)]
                masks_L.extend(masks)
            dice_L, seg_result_nonbinary, seg_result_binary, img_L, extra_pixels_L = dm.segmentation(total_loader, convert_to_HE, args.segmentation_model, 
                                                                            do_convert=args.do_convert, do_invert=args.do_invert, do_crop=args.crop, masks=masks_L)
        elif args.wang_data:
            dice_L, seg_result_binary, img_L, extra_pixels_L = dm.segmentation_wang(total_loader, convert_to_HE, args.segmentation_model,
                                                                    do_convert=args.do_convert, do_invert=args.do_invert)
        else:
            raise ValueError("No masks provided for ground truth of segmentation")
        # save dice scores as a txt file
        for idx, segmentation_model in enumerate(args.segmentation_model):
            with open(os.path.join(args.save_dir, f"dice_{segmentation_model}_{args.name}_{str(args.do_convert[idx])}.txt"), 'w') as f:
                for item in dice_L:
                    f.write("%s\n" % item[idx])
            with open(os.path.join(args.save_dir, f"extra_pixels_{segmentation_model}_{args.name}_{str(args.do_convert[idx])}.txt"), 'w') as f:
                for item in extra_pixels_L:
                    f.write("%s\n" % item[idx])
        # plot masks
        plot_masks(seg_result_binary, img_L, os.path.join(args.save_dir, "background_seg"), args.name)
    elif args.task == "feature_visualization":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dm.visualizing_features(total_loader, feature, device, args.save_dir, args.name, class_to_idx)
    elif args.task == "translation_stats":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dm.translation_stats(total_loader, convert_to_HE, device, args.save_dir, args.name, converted_path=args.converted_image_path)
    else:
        raise ValueError('Invalid task')
    print("Task completed")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='Name of the experiment')
    parser.add_argument('--hugging_face_token', type=str, default=None, 
                        help='Access token for Hugging Face Hub to use with pre-trained models (required for H-Optimus and Uni)')
    parser.add_argument('--feature_extractor', type=str, default='phikon', 
                        help='Feature extractor. Possible values: phikon, vit, h-optimus, or resnet')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Path to the model for feature extraction. If none, use ImageNet weights')
    parser.add_argument('--data_indices', type=str, nargs='+', 
                        help='Path(s) to the txt files containing indicies for data split')
    parser.add_argument('--image_path', type=str, nargs='+', help='Path(s) to the image')
    parser.add_argument('--wang_data', action='store_true', help='Use Wang data for segmentation')
    parser.add_argument('--save_dir', type=str, help='Path to save the results')
    parser.add_argument('--task', type=str, default='logistic_regression', 
                        help='Downstream task. Possible values: logistic_regression, random_forest, segmentation, feature_visualization, translation_stats')
    parser.add_argument('--coco', type=str, nargs='+', default=None, 
                        help='Path(s) to the COCO annotation file for segmentation')
    parser.add_argument('--masks', type=str, nargs='+', default=None,
                        help='Path(s) to the masks for segmentation')
    parser.add_argument('--segmentation_model', type=str, nargs="+", default=["tissuenet_cp3"], 
                        help='pre-trained models to use for segmentation')
    parser.add_argument('--do_convert', type=str2bool, default=[False], nargs='+',
                        help='If conversion is needed for each model')
    parser.add_argument('--convert', type=str, default=None, 
                        help='Convert the images to H&E using trained CycleGAN models. Put path to the model here')
    parser.add_argument('--crop', action='store_true', help='Crop the image to a quarter of its size')
    parser.add_argument('--do_invert', type=str2bool, default=[False], nargs='+',
                        help='If inversion is needed for each model')
    parser.add_argument('--invert', action='store_true', help='Invert the image')
    parser.add_argument('--load', action='store_true', help='Load pre-extracted features for downstream tasks')
    parser.add_argument('--num_epochs', type=int, default=40, help="Number of epochs, only for logistic regression")
    parser.add_argument('--do_cross_val', action='store_true', help='Do cross validation')
    parser.add_argument("--do_colour_filter", action="store_true", 
                        help="Apply colour filter to the images as a simple way to convert to H&E")
    parser.add_argument('--utom', action='store_true', help='Use UTOM for image conversion')
    parser.add_argument('--test_for_noise', action='store_true', help='Test the effect of gradually adding noise')
    parser.add_argument('--converted_image_path', type=str, default=None, 
                        help='Path to directory with converted images. Use this if you have a list of pre-converted images')
    parser.add_argument('--crop_size', type=int, default=64, help='Size to crop the image')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Program starting... Parameters:")
    print(args)
    # seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    pl.seed_everything(42, workers=True)
    # run!
    main()
