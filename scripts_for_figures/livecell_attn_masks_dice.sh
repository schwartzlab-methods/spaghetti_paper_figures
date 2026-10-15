# this will also give you the dice using the vanilla cycleGAN
python3 -u ../visualization.py --name livecell_attn_dice \
--image_path path_to_livecell_images \
--save_dir path_to_save_dir \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt \
--ssim path_to/spaghetti_model.ckpt \
--coco path_to/livecell_coco_test.json path_to/livecell_coco_train.json path_to/livecell_coco_val.json \
--crop