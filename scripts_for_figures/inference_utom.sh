#Script to run utom inference

#######! Test Script for LIVECell
python3 ../UTOM/test.py \
--dataroot path_to_utom_data/ \
--checkpoint path_to_dir_to_utom_checkpoint_dir/ \
--name UTOM --model cycle_gan --input_nc 3 --output_nc 3 --gpu_ids 0 --load_size 256 --crop_size 256 --display_winsize 512 \
--trainA_normalize 255 \
--num_test 99999999999 --results_dir path_to_save_utom_inference_results/