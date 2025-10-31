#inference for brightfield images and downstream classification

python3 -u ../downstream.py --name nocg_brightfield_patch_rf_owkin \
--task random_forest \
--do_cross_val --crop --crop_size 32 \
--feature_extractor phikon \
--save_dir path_to_save/ \
--image_path path_to_brightfield_data

python3 -u ../downstream.py --name utom_brightfield_patch_rf_owkin \
--task random_forest \
--do_cross_val --crop --crop_size 32 \
--feature_extractor phikon \
--save_dir path_to_save/ \
--image_path /home/zf2dong/schwartz-lab-ric/data/external/utom_results/final_translated/brightfield

python3 -u ../downstream.py --name nocg_brightfield_patch_rf_resnet \
--task random_forest \
--do_cross_val --crop --crop_size 32 \
--feature_extractor resnet \
--save_dir path_to_save/ \
--image_path path_to_brightfield_data

python3 -u ../downstream.py --name ssim_brightfield_patch_rf_owkin \
--task random_forest \
--do_cross_val --crop --crop_size 32 \
--feature_extractor phikon \
--save_dir path_to_save/ \
--image_path path_to_brightfield_data \
--convert path_to_checkpoint.ckpt

python3 -u ../downstream.py --name cyclegan_brightfield_patch_rf_owkin \
--task random_forest \
--do_cross_val --crop --crop_size 32 \
--feature_extractor phikon \
--save_dir path_to_save/ \
--image_path path_to_brightfield_data \
--convert /home/zf2dong/ric/results/spaghetti_results/models/originial_cyclegan/croppatch_cycleGAN_pcm_he_lr0.005_rgb_originial_cyclegan/version_0/checkpoints/epoch=99-step=9300.ckpt

python3 -u ../downstream.py --name nocg_brightfield_patch_rf_optimus \
--task random_forest \
--do_cross_val --crop --crop_size 32 \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--save_dir path_to_save/ \
--image_path path_to_brightfield_data


python3 -u ../downstream.py --name utom_brightfield_patch_rf_optimus \
--task random_forest \
--do_cross_val --crop --crop_size 32 \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--save_dir path_to_save/ \
--image_path path_to_brightfield_data

python3 -u ../downstream.py --name cyclegan_brightfield_patch_rf_optimus \
--task random_forest \
--do_cross_val --crop --crop_size 32 \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--save_dir path_to_save/ \
--image_path path_to_brightfield_data \
--convert path_to_checkpoint.ckpt

python3 -u ../downstream.py --name ssim_brightfield_patch_rf_optimus \
--task random_forest \
--do_cross_val --crop --crop_size 32 \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--save_dir path_to_save/ \
--image_path path_to_brightfield_data \
--convert path_to_checkpoint.ckpt

python3 -u ../downstream.py --name nocg_brightfield_patch_rf_uni \
--task random_forest \
--do_cross_val --crop --crop_size 32 \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--save_dir path_to_save/ \
--image_path path_to_brightfield_data \


python3 -u ../downstream.py --name utom_brightfield_patch_rf_uni \
--task random_forest \
--do_cross_val --crop --crop_size 32 \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--save_dir path_to_save/ \
--image_path path_to_brightfield_data

python3 -u ../downstream.py --name ssim_brightfield_patch_rf_uni \
--task random_forest \
--do_cross_val --crop --crop_size 32 \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--save_dir path_to_save/ \
--image_path path_to_brightfield_data \
--convert path_to_checkpoint.ckpt

python3 -u ../downstream.py --name cyclegan_brightfield_patch_rf_uni \
--task random_forest \
--do_cross_val --crop --crop_size 32 \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--save_dir path_to_save/ \
--image_path path_to_brightfield_data \
--convert path_to_checkpoint.ckpt

python3 -u ../plot_curves.py --name brightfield_class_croppatch_rf_depth2_9-1split_with_optimus_uni \
--type roc \
--no_error_bar \
--log_dir path_to_save/ssim_brightfield_patch_rf_owkin/ \
path_to_save/cyclegan_brightfield_patch_rf_owkin/ \
path_to_save/utom_brightfield_patch_rf_owkin/ \
path_to_save/nocg_brightfield_patch_rf_owkin/ \
path_to_save/ssim_brightfield_patch_rf_optimus/ \
path_to_save/cyclegan_brightfield_patch_rf_optimus/ \
path_to_save/utom_brightfield_patch_rf_optimus/ \
path_to_save/nocg_brightfield_patch_rf_optimus/ \
path_to_save/ssim_brightfield_patch_rf_uni/ \
path_to_save/cyclegan_brightfield_patch_rf_uni/ \
path_to_save/utom_brightfield_patch_rf_uni/ \
path_to_save/nocg_brightfield_patch_rf_uni/ \
path_to_save/nocg_brightfield_patch_rf_resnet/ \
--save_dir path_to_save/plots \
--labels SPAGHETTI+Phikon CycleGAN+Phikon UTOM+Phikon Phikon SPAGHETTI+H-Optimus CycleGAN+H-Optimus UTOM+H-Optimus H-Optimus SPAGHETTI+UNI CycleGAN+UNI UTOM+UNI UNI ImageNet_ResNet