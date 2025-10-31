# generate visualization for vanilla cycleGAN conversion on LIVECell testing
python3 -u ../downstream.py --name figure_2a_cycleGAN \
--task segmentation \
--image_path path_to/livecell_all_images \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt \
--save_dir path_to_save_dir \
--segmentation_model 2D_versatile_he \
--do_convert True  \
--do_invert False \
--convert path_to/vanila_cyclegan.ckpt \
--coco path_to/livecell_coco_test.json path_to/livecell_coco_train.json path_to/livecell_coco_val.json \
--crop

# generate visualization for SPAGHETTI conversion on LIVECell testing
# this will also generate the one segmentation example using 2D versatile HE model
python3 -u ../downstream.py --name figure_2a_spaghetti \
--task segmentation \
--image_path path_to/livecell_all_images \
--data_indices path_to/train_indicies.txt path_to/val_indicies.txt \
--save_dir path_to_save_dir \
--segmentation_model 2D_versatile_he \
--do_convert True  \
--do_invert False \
--convert path_to/spaghetti.ckpt \
--coco path_to/livecell_coco_test.json path_to/livecell_coco_train.json path_to/livecell_coco_val.json \
--crop