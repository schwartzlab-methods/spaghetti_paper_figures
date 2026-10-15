# train classifiers on livecell dataset
srun python3 -u ../downstream.py --name nocg_LIVECELL_patch_rf_owkin \
--task random_forest \
--crop \
--do_cross_val \
--feature_extractor phikon \
--save_dir path_to_save_dir \
--image_path path_to_livecell_images \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt

srun python3 -u ../downstream.py --name nocg_LIVECELL_patch_rf_resnet \
--task random_forest \
--crop \
--do_cross_val \
--feature_extractor resnet \
--save_dir path_to_save_dir \
--image_path path_to_livecell_images \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt

srun python3 -u ../downstream.py --name ssim_LIVECELL_patch_rf_owkin \
--task random_forest \
--crop \
--do_cross_val \
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
--do_cross_val \
--save_dir path_to_save_dir \
--image_path path_to_livecell_images \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt

srun python3 -u ../downstream.py --name ssim_LIVECELL_patch_rf_optimus \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--do_cross_val \
--save_dir path_to_save_dir \
--image_path path_to_livecell_images \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt \
--convert path_to/spaghetti_model.ckpt

srun python3 -u ../downstream.py --name nocg_LIVECELL_patch_rf_uni \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--do_cross_val \
--save_dir path_to_save_dir \
--image_path path_to_livecell_images \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt

srun python3 -u ../downstream.py --name ssim_LIVECELL_patch_rf_uni \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--do_cross_val \
--save_dir path_to_save_dir \
--image_path path_to_livecell_images \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt \
--convert path_to/spaghetti_model.ckpt