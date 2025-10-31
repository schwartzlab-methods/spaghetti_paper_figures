# this will also give you the visualization using the vanilla cycleGAN
python3 -u ../visualization.py --name livecell_attn \
--image_path path_to_livecell_images \
--save_dir path_to_save_dir \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt \
--ssim path_to/spaghetti_model.ckpt \
--crop