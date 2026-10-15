#DCI inference

python3 -u ../downstream.py --name nocg_dic_patch_rf_owkin \
--task random_forest \
--do_cross_val \
--feature_extractor phikon \
--save_dir path_to_data/ \
--image_path path_to_alfi

python3 -u ../downstream.py --name utom_dic_patch_rf_owkin \
--task random_forest \
--do_cross_val \
--feature_extractor phikon \
--save_dir path_to_data/ \
--image_path path_to_translated_alfi_by_utom

python3 -u ../downstream.py --name nocg_dic_patch_rf_resnet \
--task random_forest \
--do_cross_val \
--feature_extractor resnet \
--save_dir path_to_data/ \
--image_path path_to_alfi

python3 -u ../downstream.py --name ssim_dic_patch_rf_owkin \
--task random_forest \
--do_cross_val \
--feature_extractor phikon \
--save_dir path_to_data/ \
--image_path path_to_alfi \
--convert path_to_checkpoint.ckpt

python3 -u ../downstream.py --name cyclegan_dic_patch_rf_owkin \
--task random_forest \
--do_cross_val \
--feature_extractor phikon \
--save_dir path_to_data/ \
--image_path path_to_alfi \
--convert path_to_checkpoint.ckpt

python3 -u ../downstream.py --name nocg_dic_patch_rf_optimus \
--task random_forest \
--do_cross_val \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--save_dir path_to_data/ \
--image_path path_to_alfi


python3 -u ../downstream.py --name utom_dic_patch_rf_optimus \
--task random_forest \
--do_cross_val \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--save_dir path_to_data/ \
--image_path path_to_translated_alfi_by_utom

python3 -u ../downstream.py --name cyclegan_dic_patch_rf_optimus \
--task random_forest \
--do_cross_val \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--save_dir path_to_data/ \
--image_path path_to_alfi \
--convert path_to_checkpoint.ckpt

python3 -u ../downstream.py --name ssim_dic_patch_rf_optimus \
--task random_forest \
--do_cross_val \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--save_dir path_to_data/ \
--image_path path_to_alfi \
--convert path_to_checkpoint.ckpt

python3 -u ../downstream.py --name nocg_dic_patch_rf_uni \
--task random_forest \
--do_cross_val \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--save_dir path_to_data/ \
--image_path path_to_alfi \


python3 -u ../downstream.py --name utom_dic_patch_rf_uni \
--task random_forest \
--do_cross_val \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--save_dir path_to_data/ \
--image_path path_to_translated_alfi_by_utom

python3 -u ../downstream.py --name ssim_dic_patch_rf_uni \
--task random_forest \
--do_cross_val \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--save_dir path_to_data/ \
--image_path path_to_alfi \
--convert path_to_checkpoint.ckpt

python3 -u ../downstream.py --name cyclegan_dic_patch_rf_uni \
--task random_forest \
--do_cross_val \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--save_dir path_to_data/ \
--image_path path_to_alfi \
--convert path_to_checkpoint.ckpt

python3 -u ../plot_curves.py --name dic_class_croppatch_rf_depth2_9-1split_with_optimus_uni \
--type roc \
--no_error_bar \
--log_dir path_to_data/ssim_dic_patch_rf_owkin/ \
path_to_data/cyclegan_dic_patch_rf_owkin/ \
path_to_data/utom_dic_patch_rf_owkin/ \
path_to_data/nocg_dic_patch_rf_owkin/ \
path_to_data/ssim_dic_patch_rf_optimus/ \
path_to_data/cyclegan_dic_patch_rf_optimus/ \
path_to_data/utom_dic_patch_rf_optimus/ \
path_to_data/nocg_dic_patch_rf_optimus/ \
path_to_data/ssim_dic_patch_rf_uni/ \
path_to_data/cyclegan_dic_patch_rf_uni/ \
path_to_data/utom_dic_patch_rf_uni/ \
path_to_data/nocg_dic_patch_rf_uni/ \
path_to_data/nocg_dic_patch_rf_resnet/ \
--save_dir path_to_data/plots \
--labels SPAGHETTI+Phikon CycleGAN+Phikon UTOM+Phikon Phikon SPAGHETTI+H-Optimus CycleGAN+H-Optimus UTOM+H-Optimus H-Optimus SPAGHETTI+UNI CycleGAN+UNI UTOM+UNI UNI ImageNet_ResNet