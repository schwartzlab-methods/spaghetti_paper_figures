# generate the box plot
python3 -u ../plot_curves.py --name figure_2c \
--type box \
--log_dir path_to_each_txt_log_file_seperated_by_space \
--save_dir path_to_save_dir \
--labels SPAGHETTI+tissuenet_cp3 CycleGAN+tissuenet_cp3 tissuenet_cp3_Only cyto3_only SPAGHETTI+2D_versatile_he CycleGAN+2D_versatile_he 2D_versatile_he_only

