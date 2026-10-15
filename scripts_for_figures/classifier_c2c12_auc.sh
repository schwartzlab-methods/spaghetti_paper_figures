# plot auc curves for c2c12 classifier
# This will also generate an ROC curve for all (combined Supplemental Figure 2)
python3 -u ../plot_curves.py --name c2c12_class_croppatch_rf_depth2_9-1split \
--type roc \
--no_error_bar \
--log_dir path_to_save_dir/ssim_c2c12_patch_rf_owkin/ \
path_to_save_dir/filter_c2c12_patch_rf_owkin/ \
path_to_save_dir/nocg_c2c12_patch_rf_owkin/ \
path_to_save_dir/ssim_c2c12_patch_rf_optimus/ \
path_to_save_dir/filter_c2c12_patch_rf_optimus/ \
path_to_save_dir/nocg_c2c12_patch_rf_optimus/ \
path_to_save_dir/ssim_c2c12_patch_rf_uni/ \
path_to_save_dir/filter_c2c12_patch_rf_uni/ \
path_to_save_dir/nocg_c2c12_patch_rf_uni/ \
path_to_save_dir/nocg_c2c12_patch_rf_resnet/ \
--save_dir path_to_save_dir \
--labels SPAGHETTI+Phikon Filter+Phikon Phikon SPAGHETTI+H-Optimus filter+H-Optimus H-Optimus SPAGHETTI+UNI filter+UNI UNI ImageNet_ResNet