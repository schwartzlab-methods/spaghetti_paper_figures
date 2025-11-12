python3 -u ../train_classifier_he.py \
--feature \
path_to/visualization_pannuke_livecell_convert_h-optimus_features.npz \
--output save_path/ \
--name classifier_he_convert_h-optimus

python3 -u ../train_classifier_he.py \
--feature \
path_to/visualization_pannuke_livecell_NOconvert_h-optimus_features.npz \
--output save_path/ \
--name classifier_he_NOconvert_h-optimus

python3 ../plot_auc.py \
--csv_files \
path_to/pcm_class_auc_data_classifier_he_convert_h-optimus.csv \
path_to/pcm_class_auc_data_classifier_he_NOconvert_h-optimus.csv \
--labels with_spaghetti without_spaghetti --save_path save_path