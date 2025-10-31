#plot the box plot for pearson correlation


# make inferences, run this for every model you want to test
# leave the --convert flag empty if you do not want to use the spaghetti model
python3 -u ../gfp_inference.py \
--name name_of_model --russian_dataset \
--feature_extractor your_extractor \
--hugging_face_token your_hugging_face_token \
--image_path path_to_viability_dataset \
--save_dir path_to_save \
--convert path_to_spaghetti \
--data_indices path_to_data_indices_generated_during_training \
--method lr-cv

# plot the box plot for pearson correlation
python3 -u ../plot_curves.py --name r_values \
--type box --file_type npy \
--log_dir paths_to_numpy_files \
--save_dir save_path \
--labels retrained_SPAGHETTI+Phikon-v2 old_sp+phikon-v2 Phikon-v2_Only retrained_SPAGHETTI+H-Optimus old_sp+h-optimus H-Optimus_Only retrained_SPAGHETTI+UNI old_sp+uni UNI_Only Resnet

# stats test
python3 -u ../stats_test.py \
--f1 numpy_sample_1_path \
--f2 numpy_sample_2_path \
--test permutation --npy