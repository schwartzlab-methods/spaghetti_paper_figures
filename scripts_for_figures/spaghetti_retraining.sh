# retraining SPAGHETTI
# this script works with multiple GPUs

srun python3 -u ../train_cycleGAN.py \
--name retrain_spaghetti \
--path_1 path_to_cell_viability \
--path_2 path_to_pannuke \
--save_dir path_to_save \
--lr 0.0005 \
--epochs 100 \
--RGB --gfp_data \
--weights 1.0 10.0 5.0 5.0

# plot the loss
python3 -u ../plot_loss.py \
--losses path_to_csvs_genertated_during_training \
--save_path save_directory