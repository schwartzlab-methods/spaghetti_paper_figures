# this will ALSO give you the visualization using the vanilla cycleGAN
python3 -u ../visualization.py --name c2c12_attn \
--image_path path_to_c2c12_images \
--save_dir path_to_save_dir \
--ssim path_to/spaghetti_model.ckpt \
--no_ssim path_to/vanilla_cyclegan_model.ckpt \
--crop