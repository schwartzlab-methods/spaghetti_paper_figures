# script to run noise robustness experiments for LIVECell datasets

#! LIVECell
# spaghetti
python3 -u ../downstream.py --name ssim_LIVECELL_patch_rf_owkin \
--task random_forest --test_for_noise \
--feature_extractor phikon \
--num_epochs 100 \
--save_dir path_to_save/ \
--image_path path_to/livecell_test_images/ path_to/livecell_train_val_images/ \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt \
--convert path_to_spaghetti.ckpt \
--crop --crop_size 128

python3 -u ../downstream.py --name ssim_LIVECELL_patch_rf_optimus \
--task random_forest --test_for_noise \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--num_epochs 100 \
--save_dir path_to_save/ \
--image_path path_to/livecell_test_images/ path_to/livecell_train_val_images/ \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt \
--convert path_to_spaghetti.ckpt \
--crop --crop_size 128

python3 -u ../downstream.py --name ssim_LIVECELL_patch_rf_uni \
--task random_forest --test_for_noise \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--num_epochs 100 \
--save_dir path_to_save/ \
--image_path path_to/livecell_test_images/ path_to/livecell_train_val_images/ \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt \
--convert path_to_spaghetti.ckpt \
--crop --crop_size 128

# no convert
python3 -u ../downstream.py --name nocg_LIVECELL_patch_rf_owkin \
--task random_forest --test_for_noise \
--feature_extractor phikon \
--num_epochs 100 \
--save_dir path_to_save/ \
--image_path path_to/livecell_test_images/ path_to/livecell_train_val_images/ \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt \
--crop --crop_size 128

python3 -u ../downstream.py --name nocg_LIVECELL_patch_rf_optimus \
--task random_forest --test_for_noise \
--hugging_face_token your_hugging_face_token \
--feature_extractor h-optimus \
--num_epochs 100 \
--save_dir path_to_save/ \
--image_path path_to/livecell_test_images/ path_to/livecell_train_val_images/ \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt \
--crop --crop_size 128

python3 -u ../downstream.py --name nocg_LIVECELL_patch_rf_uni \
--task random_forest --test_for_noise \
--hugging_face_token your_hugging_face_token \
--feature_extractor uni \
--num_epochs 100 \
--save_dir path_to_save/ \
--image_path path_to/livecell_test_images/ path_to/livecell_train_val_images/ \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt \
--crop --crop_size 128

#! Noise experiment for LIVECell
DIR_LIVECELL=path_to_save

python3 -u ../line_chart_plot.py \
--data_y \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_optimus/*_0*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_optimus/*_1*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_optimus/*_2*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_optimus/*_3*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_optimus/*_4*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_optimus/*_5*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_optimus/*_6*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_optimus/*_7*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_optimus/*_8*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_optimus/*_9*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_optimus/*_10*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_owkin/*_0*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_owkin/*_1*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_owkin/*_2*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_owkin/*_3*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_owkin/*_4*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_owkin/*_5*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_owkin/*_6*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_owkin/*_7*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_owkin/*_8*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_owkin/*_9*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_owkin/*_10*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_uni/*_0*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_uni/*_1*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_uni/*_2*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_uni/*_3*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_uni/*_4*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_uni/*_5*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_uni/*_6*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_uni/*_7*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_uni/*_8*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_uni/*_9*.txt \
${DIR_LIVECELL}/nocg_LIVECELL_patch_rf_uni/*_10*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_optimus/*_0*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_optimus/*_1*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_optimus/*_2*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_optimus/*_3*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_optimus/*_4*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_optimus/*_5*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_optimus/*_6*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_optimus/*_7*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_optimus/*_8*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_optimus/*_9*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_optimus/*_10*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_owkin/*_0*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_owkin/*_1*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_owkin/*_2*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_owkin/*_3*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_owkin/*_4*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_owkin/*_5*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_owkin/*_6*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_owkin/*_7*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_owkin/*_8*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_owkin/*_9*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_owkin/*_10*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_uni/*_0*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_uni/*_1*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_uni/*_2*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_uni/*_3*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_uni/*_4*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_uni/*_5*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_uni/*_6*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_uni/*_7*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_uni/*_8*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_uni/*_9*.txt \
${DIR_LIVECELL}/ssim_LIVECELL_patch_rf_uni/*_10*.txt \
--data_x \
path_to_save/noises_added_x.txt \
path_to_save/noises_added_x.txt \
path_to_save/noises_added_x.txt \
path_to_save/noises_added_x.txt \
path_to_save/noises_added_x.txt \
path_to_save/noises_added_x.txt \
--exp_types Raw Raw SPAGHETTI SPAGHETTI SPAGHETTI Raw \
--exp_names \
Optimus_LIVECell Owkin_LIVECell UNI_LIVECell Optimus_LIVECell Owkin_LIVECell UNI_LIVECell \
--x_title "Number of Added Gaussian Noise" --y_title AUC \
--save path_to_save/ --name noise_addition_std
