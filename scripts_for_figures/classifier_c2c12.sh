# train classifiers on c2c12 dataset
srun python3 -u ../downstream.py --name nocg_c2c12_patch_rf_owkin \
--task random_forest \
--crop \
--feature_extractor phikon \
--save_dir path_to_save_dir \
--image_path path_to_c2c12_images \
--do_cross_val

srun python3 -u ../downstream.py --name nocg_c2c12_patch_rf_resnet \
--task random_forest \
--crop \
--feature_extractor resnet \
--save_dir path_to_save_dir \
--image_path path_to_c2c12_images \
--do_cross_val

srun python3 -u ../downstream.py --name ssim_c2c12_patch_rf_owkin \
--task random_forest \
--crop \
--feature_extractor phikon \
--save_dir path_to_save_dir \
--image_path path_to_c2c12_images \
--convert path_to/spaghetti_model.ckpt \
--do_cross_val

srun python3 -u ../downstream.py --name nocg_c2c12_patch_rf_optimus \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--save_dir path_to_save_dir \
--image_path path_to_c2c12_images \
--do_cross_val

srun python3 -u ../downstream.py --name ssim_c2c12_patch_rf_optimus \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--num_epochs 100 \
--save_dir path_to_save_dir \
--image_path path_to_c2c12_images \
--convert path_to/spaghetti_model.ckpt \
--do_cross_val

srun python3 -u ../downstream.py --name nocg_c2c12_patch_rf_uni \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--save_dir path_to_save_dir \
--image_path path_to_c2c12_images \
--do_cross_val

srun python3 -u ../downstream.py --name ssim_c2c12_patch_rf_uni \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--save_dir path_to_save_dir \
--image_path path_to_c2c12_images \
--convert path_to/spaghetti_model.ckpt \
--do_cross_val

srun python3 -u ../downstream.py --name filter_c2c12_patch_rf_uni \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--save_dir path_to_save_dir \
--image_path path_to_c2c12_images \
--do_cross_val --do_colour_filter

srun python3 -u ../downstream.py --name filter_c2c12_patch_rf_owkin \
--task random_forest \
--crop \
--feature_extractor phikon \
--save_dir path_to_save_dir \
--image_path path_to_c2c12_images \
--do_cross_val --do_colour_filter

srun python3 -u ../downstream.py --name filter_c2c12_patch_rf_optimus \
--task random_forest \
--crop \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--save_dir path_to_save_dir \
--image_path path_to_c2c12_images \
--do_cross_val --do_colour_filter