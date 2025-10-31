#Script to compute the pair-wise distance between features extracted from images of LIVECell and PanNuke

EXTRACTOR=h-optimus

python3 -u ../downstream.py \
--task feature_visualization \
--image_path \
path_to/livecell_test_images \
path_to/livecell_train_val_images \
path_to/pannuke_patch \
--name visualization_pannuke_livecell_convert_${EXTRACTOR} \
--hugging_face_token your_hugging_face_token \
--feature_extractor ${EXTRACTOR} \
--save_dir path_to_save/ \
--convert path_to_spaghetti_model.ckpt

python3 -u ../downstream.py \
--task feature_visualization \
--image_path \
path_to/livecell_test_images \
path_to/livecell_train_val_images \
path_to/pannuke_patch \
--name visualization_pannuke_livecell_NOconvert_${EXTRACTOR} \
--hugging_face_token your_hugging_face_token \
--feature_extractor ${EXTRACTOR} \
--save_dir path_to_save/

# plots
python3 -u ../compute_pca_distance.py \
--features_distance_matrix_path \
path_to_save/visualization_pannuke_livecell_convert_h-optimus_pairwise_distances.npz \
path_to_save/visualization_pannuke_livecell_NOconvert_h-optimus_pairwise_distances.npz \
--save_dir path_to_save/