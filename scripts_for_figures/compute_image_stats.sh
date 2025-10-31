#Compute the quality of the translation

####* LIVECell
python3 -u ../downstream.py \
--task translation_stats \
--image_path \
path_to/livecell_test_images \
path_to/livecell_train_val_images \
--name translation_stats_livecell_convert_spaghetti \
--convert path_to_checkpoint.ckpt \
--save_dir path_to_save \
--crop

python3 -u ../downstream.py \
--task translation_stats \
--image_path \
path_to/livecell_test_images \
path_to/livecell_train_val_images \
--name translation_stats_livecell_convert_cyclegan \
--convert path_to_checkpoint.ckpt \
--save_dir path_to_save \
--crop

python3 -u ../downstream.py \
--task translation_stats \
--image_path \
path_to/livecell_test_images \
path_to/livecell_train_val_images \
--name translation_stats_livecell_convert_utom \
--save_dir path_to_save \
--converted_image_path path_to/utom_results_final_translated_livecell

# plots
python3 -u ../plot_curves.py --name livecell_image_metrics_spaghetti_cyclegan \
--type box \
--log_dir path_to_save/translation_stats_livecell_convert_spaghetti_lpips.txt \
path_to_save/translation_stats_livecell_convert_cyclegan_lpips.txt \
path_to_save/translation_stats_livecell_convert_utom_lpips.txt \
path_to_save/translation_stats_livecell_convert_spaghetti_ssim.txt \
path_to_save/translation_stats_livecell_convert_cyclegan_ssim.txt \
path_to_save/translation_stats_livecell_convert_utom_ssim.txt \
--save_dir path_to_save/ \
--labels SPAGHETTI_LPIPS Cyclegan_LPIPS UTOM_LPIPS SPAGHETTI_SSIM Cyclegan_SSIM UTOM_SSIM

python3 -u ../plot_curves.py --name livecell_image_metrics_spaghetti_cyclegan_psnr \
--type box \
--log_dir path_to_save/translation_stats_livecell_convert_spaghetti_psnr.txt \
path_to_save/translation_stats_livecell_convert_cyclegan_psnr.txt \
path_to_save/translation_stats_livecell_convert_utom_psnr.txt \
--save_dir path_to_save/ \
--labels SPAGHETTI_PSNR Cyclegan_PSNR UTOM_PSNR

###* C2C12
python3 -u ../downstream.py \
--task translation_stats \
--image_path \
path_to/C2C12_pcm \
--name translation_stats_c2c12_convert_spaghetti \
--convert path_to_checkpoint.ckpt \
--save_dir path_to_save \
--invert

python3 -u ../downstream.py \
--task translation_stats \
--image_path \
/home/zf2dong/schwartz-lab-ric/data/external/C2C12_pcm \
--name translation_stats_c2c12_convert_cyclegan \
--convert path_to_checkpoint.ckpt \
--save_dir path_to_save \
--invert

python3 -u ../downstream.py \
--task translation_stats \
--image_path \
/home/zf2dong/schwartz-lab-ric/data/external/C2C12_pcm \
--name translation_stats_c2c12_convert_utom \
--save_dir path_to_save \
--converted_image_path path_to/utom_results_final_translated_c2c12

# plots
python3 -u ../plot_curves.py --name c2c12_image_metrics_spaghetti_cyclegan \
--type box \
--log_dir path_to_save/translation_stats_c2c12_convert_spaghetti_lpips.txt \
path_to_save/translation_stats_c2c12_convert_cyclegan_lpips.txt \
path_to_save/translation_stats_c2c12_convert_utom_lpips.txt \
path_to_save/translation_stats_c2c12_convert_spaghetti_ssim.txt \
path_to_save/translation_stats_c2c12_convert_cyclegan_ssim.txt \
path_to_save/translation_stats_c2c12_convert_utom_ssim.txt \
--save_dir path_to_save/ \
--labels SPAGHETTI_LPIPS Cyclegan_LPIPS UTOM_LPIPS SPAGHETTI_SSIM Cyclegan_SSIM UTOM_SSIM

python3 -u ../plot_curves.py --name c2c12_image_metrics_spaghetti_cyclegan_psnr \
--type box \
--log_dir path_to_save/translation_stats_c2c12_convert_spaghetti_psnr.txt \
path_to_save/translation_stats_c2c12_convert_cyclegan_psnr.txt \
path_to_save/translation_stats_c2c12_convert_utom_psnr.txt \
--save_dir path_to_save/ \
--labels SPAGHETTI_PSNR Cyclegan_PSNR UTOM_PSNR
