## Generate the attention maps for the viability model

python3 -u ../visualization.py --name viability_nospaghetti \
--gfp_data_russian \
--image_path path_to_viability_dataset \
--save_dir path_to_save

python3 -u ../visualization.py --name viability_original \
--gfp_data_russian \
--image_path path_to_viability_dataset \
--save_dir path_to_save \
--ssim path_to_original.ckpt

python3 -u ../visualization.py --name viability_retrained \
--gfp_data_russian \
--image_path path_to_viability_dataset \
--save_dir path_to_save \
--ssim path_to_retrained.ckpt 