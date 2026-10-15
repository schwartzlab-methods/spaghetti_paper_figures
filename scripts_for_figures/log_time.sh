# we first generate the txt files for box plots
# this will automatically adapt for GPU or CPU usage
python3 -u ../log_time_and_memory.py --name c2c12 \
--image_path path_to_c2c12_images \
--save_dir path_to_save_dir \
--hugging_face_token your_hugging_face_token \
--convert path_to/spaghetti_model.ckpt

# we then plot the box plots
python3 -u ../plot_curves.py --name run_time \
--type bar \
--log_dir path_to_save/c2c12_Phikon-v2_time.txt \
path_to_save/c2c12_Phikon-v2_no_convert_time.txt \
path_to_save/c2c12_H-optimus-0_no_convert_time.txt \
path_to_save/c2c12_H-optimus-0_time.txt \
path_to_save/c2c12_UNI_time.txt \
path_to_save/c2c12_UNI_no_convert_time.txt \
--save_dir path_to_save \
--labels SPAGHETTI+Phikon-v2 Phikon-v2_Only SPAGHETTI+H-Optimus H-Optimus_Only SPAGHETTI+UNI UNI_Only

python3 -u ../plot_curves.py --name memory_usage \
--type bar \
--log_dir path_to_save/c2c12_Phikon-v2_memory.txt \
path_to_save/c2c12_Phikon-v2_no_convert_memory.txt \
path_to_save/c2c12_H-optimus-0_no_convert_memory.txt \
path_to_save/c2c12_H-optimus-0_memory.txt \
path_to_save/c2c12_UNI_memory.txt \
path_to_save/c2c12_UNI_no_convert_memory.txt \
--save_dir path_to_save \
--labels SPAGHETTI+Phikon-v2 Phikon-v2_Only SPAGHETTI+H-Optimus H-Optimus_Only SPAGHETTI+UNI UNI_Only