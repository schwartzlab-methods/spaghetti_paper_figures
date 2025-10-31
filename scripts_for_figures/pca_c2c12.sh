# we first genearate the features and save them in numpy files
python3 -u ../visualization.py --name features_c2c12 \
--image_path path_to/c2c12_images \
--save_dir path_to_save_dir \
--invert \
--ssim path_to/spaghetti_model.ckpt \
--no_ssim path_to/vanilla_cyclegan_model.ckpt \
--crop

# we then run PCA on the genearted numpy files
python3 -u ../dimension_reduction.py \
--path_1 path_to/Phikon_only.npy \
--path_2 path_to/SPAGHETTI.npy \
--path_label path_to/cell_type.npy \
--extractor Phikon-v2 \
--exp_name c2c12_treatment \
--save_dir path_to_save_dir