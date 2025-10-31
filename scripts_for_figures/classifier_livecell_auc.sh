# plot auc curves for live cell classifier
# This will also generate an ROC curve for all (combined Supplemental Figure 1)
python3 -u ../plot_curves.py --name LIVECELL_class_croppatch_rf_depth2_9-1split \
--type roc \
--no_error_bar \
--log_dir path_to_save_dir/ssim_LIVECELL_patch_rf_owkin/ \
path_to_save_dir/nocg_LIVECELL_patch_rf_owkin/ \
path_to_save_dir/ssim_LIVECELL_patch_rf_optimus/ \
path_to_save_dir/nocg_LIVECELL_patch_rf_optimus/ \
path_to_save_dir/ssim_LIVECELL_patch_rf_uni/ \
path_to_save_dir/nocg_LIVECELL_patch_rf_uni/ \
path_to_save_dir/nocg_LIVECELL_patch_rf_resnet/ \
--save_dir path_to_save_dir \
--labels SPAGHETTI+Phikon Phikon SPAGHETTI+H-Optimus H-Optimus SPAGHETTI+UNI UNI ImageNet_ResNet