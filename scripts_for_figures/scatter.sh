# plot scatter for SPAGHETTI retraining

python3 -u ../process_gfp_datafraom.py \
--dir path_to_viability_csv_directory

python3 -u ../plot_scatter.py \
--f1 csv_for_h-optimus_only \
--save_dir save_path \
--name scatter_h-optimus_only

python3 -u ../plot_scatter.py \
--f1 csv_for_spaghetti_and_h-optimus \
--save_dir save_path \
--name scatter_spaghetti_and_h-optimus

# supplemental fig 5 - the MSE
python3 -u ../compute_mse.py \
--inference paths_to_inference_csv_files \
--inference_name labels_for_each_inference \
--save_dir save_path