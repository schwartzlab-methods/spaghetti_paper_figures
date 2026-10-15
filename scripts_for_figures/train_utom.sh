#Script to train utom

python3 -u ../UTOM/train.py \
--dataroot path_to_utom/ \
--checkpoint path_to_dir_to_save_utom/ \
--trainA_normalize 225 \
--n_epochs 50 --n_epochs_decay 50 \
--name UTOM --model cycle_gan --input_nc 3 --output_nc 3 --lambda_identity 10 --gpu_ids 0 --load_size 256 --crop_size 256 --display_winsize 256 \
--lambda_A 30.0 --lambda_B 30.0 --save_epoch_freq 5 \
--threshold_A 118 --threshold_B 174