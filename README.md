# SPAGHETTI Figures
This repository contains the code to reproduce the plots used in the SPAGHETTI paper.

The main SPAGHETTI program can be found at: https://github.com/schwartzlab-methods/spaghetti 

Read the paper at: https://www.biorxiv.org/content/10.1101/2025.08.27.672636v1.abstract

## Downloading and Pre-processing Data
Please first download the datasets used to train and validate the model:
- LIVECell (used for both training and validation): http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip
- PanNuke (used for training only): https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
- C2C12 Culture Media (used for validation only): https://osf.io/ysaq2/ 
- DIC ALFI dataset (used for validation only): https://springernature.figshare.com/articles/dataset/ALFI_dataset_final_/23798451?backTo=/collections/ALFI_Cell_cycle_phenotype_annotations_of_label-free_time-lapse_imaging_data_from_cultured_human_cells/6436958 
- Brightfield dataset (used for validation only) from `data/`: https://zenodo.org/records/8415315 

Please ensure that all images are **organized by the cell type or culture media**. For example, your LIVECell folder should look something like this after your pre-processing:
```
livecell/
    - A172/
        - all_images_of_A172_cells
    - BT474/
        - all_images_of_BT474_cells
    ...
```

In addition, please also download the cell viability dataset, which can be used as is. You can download it from https://figshare.com/articles/dataset/VirtualStaining_Dataset/21971558/1?file=38982755

## Running Scripts
All scripts are located at `script_for_figures/`. Before running the scripts, since there will be a lot of outputs and inputs being generated, **make sure you go to each bash script file and change the path to the corresponding inputs and outputs based on the comments in the bash script**. Please also read the argument for each `.py` files to be run to understand what argument is required.
