# train classifiers on c2c12 dataset, varying training size
srun python3 -u ../downstream.py --name nocg_c2c12_patch_rf_owkin \
--task random_forest \
--crop \
--feature_extractor phikon \
--save_dir path_to_save_dir \
--image_path path_to_c2c12_images \

srun python3 -u ../downstream.py --name ssim_c2c12_patch_rf_owkin \
--task random_forest \
--crop \
--feature_extractor phikon \
--save_dir path_to_save_dir \
--image_path path_to_c2c12_images \
--convert path_to/spaghetti_model.ckpt \

srun python3 -u ../downstream.py --name nocg_c2c12_patch_rf_optimus \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--save_dir path_to_save_dir \
--image_path path_to_c2c12_images \

srun python3 -u ../downstream.py --name ssim_c2c12_patch_rf_optimus \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--num_epochs 100 \
--save_dir path_to_save_dir \
--image_path path_to_c2c12_images \
--convert path_to/spaghetti_model.ckpt \

srun python3 -u ../downstream.py --name nocg_c2c12_patch_rf_uni \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--save_dir path_to_save_dir \
--image_path path_to_c2c12_images \

srun python3 -u ../downstream.py --name ssim_c2c12_patch_rf_uni \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--save_dir path_to_save_dir \
--image_path path_to_c2c12_images \
--convert path_to/spaghetti_model.ckpt \

# train classifiers on livecell dataset, varying training size
srun python3 -u ../downstream.py --name nocg_LIVECELL_patch_rf_owkin \
--task random_forest \
--crop \
--feature_extractor phikon \
--save_dir path_to_save_dir \
--image_path path_to_livecell_images \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt

srun python3 -u ../downstream.py --name ssim_LIVECELL_patch_rf_owkin \
--task random_forest \
--crop \
--feature_extractor phikon \
--save_dir path_to_save_dir \
--image_path path_to_livecell_images \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt \
--convert path_to/spaghetti_model.ckpt

srun python3 -u ../downstream.py --name nocg_LIVECELL_patch_rf_optimus \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--save_dir path_to_save_dir \
--image_path path_to_livecell_images \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt

srun python3 -u ../downstream.py --name ssim_LIVECELL_patch_rf_optimus \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--save_dir path_to_save_dir \
--image_path path_to_livecell_images \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt \
--convert path_to/spaghetti_model.ckpt

srun python3 -u ../downstream.py --name nocg_LIVECELL_patch_rf_uni \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--save_dir path_to_save_dir \
--image_path path_to_livecell_images \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt

srun python3 -u ../downstream.py --name ssim_LIVECELL_patch_rf_uni \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--save_dir path_to_save_dir \
--image_path path_to_livecell_images \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt \
--convert path_to/spaghetti_model.ckpt

# finally we generate the scatter plot
python3 -u ../line_chart_plot.py \
--data_y path_to_save/nocg_LIVECELL_patch_rf_optimus/auc_nocg_LIVECELL_patch_rf_optimus.txt \
path_to_save/ssim_LIVECELL_patch_rf_optimus/auc_ssim_LIVECELL_patch_rf_optimus.txt \
path_to_save/nocg_LIVECELL_patch_rf_owkin/auc_nocg_LIVECELL_patch_rf_owkin.txt \
path_to_save/ssim_LIVECELL_patch_rf_owkin/auc_ssim_LIVECELL_patch_rf_owkin.txt \
path_to_save/nocg_LIVECELL_patch_rf_uni/auc_nocg_LIVECELL_patch_rf_uni.txt \
path_to_save/ssim_LIVECELL_patch_rf_uni/auc_ssim_LIVECELL_patch_rf_uni.txt \
path_to_save/nocg_c2c12_patch_rf_optimus/auc_nocg_c2c12_patch_rf_optimus.txt \
path_to_save/ssim_c2c12_patch_rf_optimus/auc_ssim_c2c12_patch_rf_optimus.txt \
path_to_save/nocg_c2c12_patch_rf_owkin/auc_nocg_c2c12_patch_rf_owkin.txt \
path_to_save/ssim_c2c12_patch_rf_owkin/auc_ssim_c2c12_patch_rf_owkin.txt \
path_to_save/nocg_c2c12_patch_rf_uni/auc_nocg_c2c12_patch_rf_uni.txt \
path_to_save/ssim_c2c12_patch_rf_uni/auc_ssim_c2c12_patch_rf_uni.txt \
--data_x path_to_save/ssim_LIVECELL_patch_rf_uni/percent_train_ssim_LIVECELL_patch_rf_uni.txt \
path_to_save/ssim_LIVECELL_patch_rf_uni/percent_train_ssim_LIVECELL_patch_rf_uni.txt \
path_to_save/ssim_LIVECELL_patch_rf_uni/percent_train_ssim_LIVECELL_patch_rf_uni.txt \
path_to_save/ssim_LIVECELL_patch_rf_uni/percent_train_ssim_LIVECELL_patch_rf_uni.txt \
path_to_save/ssim_LIVECELL_patch_rf_uni/percent_train_ssim_LIVECELL_patch_rf_uni.txt \
path_to_save/ssim_LIVECELL_patch_rf_uni/percent_train_ssim_LIVECELL_patch_rf_uni.txt \
path_to_save/ssim_LIVECELL_patch_rf_uni/percent_train_ssim_LIVECELL_patch_rf_uni.txt \
path_to_save/ssim_LIVECELL_patch_rf_uni/percent_train_ssim_LIVECELL_patch_rf_uni.txt \
path_to_save/ssim_LIVECELL_patch_rf_uni/percent_train_ssim_LIVECELL_patch_rf_uni.txt \
path_to_save/ssim_LIVECELL_patch_rf_uni/percent_train_ssim_LIVECELL_patch_rf_uni.txt \
path_to_save/ssim_LIVECELL_patch_rf_uni/percent_train_ssim_LIVECELL_patch_rf_uni.txt \
path_to_save/ssim_LIVECELL_patch_rf_uni/percent_train_ssim_LIVECELL_patch_rf_uni.txt \
--exp_types Raw_Image SPAGHETTI Raw_Image SPAGHETTI Raw_Image SPAGHETTI Raw_Image SPAGHETTI Raw_Image SPAGHETTI Raw_Image SPAGHETTI \
--exp_names LIVECell+H-Optimus LIVECell+H-Optimus LIVECell+Phikon LIVECell+Phikon LIVECell+UNI LIVECell+UNI C2C12+H-Optimus C2C12+H-Optimus C2C12+Phikon C2C12+Phikon C2C12+UNI C2C12+UNI \
--save path_to_save --x_title Percent_Training_Data --y_title AUC