srun python3 -u ../train_cycleGAN.py \
--name spaghetti_model \
--path_1 path_to_livecell_patches \
--path_2 path_to_pannuke_patches \
--save_dir path_to_save_directory \
--lr 0.0005 \
--epochs 100 \
--crop \
--RGB \
--weights 1.0 10.0 0.5 1.0