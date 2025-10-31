# script for tunning SSIM weight for SPAGHETTI model on LIVECell dataset

# repreat this first for all ssim weights you want to test: 0.5, 1, 1.5, 2, 3, 10
weight=0.5
python3 -u ../downstream.py \
--task translation_stats \
--image_path \
path_to/livecell_test_images \
path_to/livecell_train_val_images \
--name translation_stats_livecell_crop_${weight}_ssim \
--convert path_to_checkpoint.ckpt \
--save_dir path_to_save/ \
--crop

####! Box plot for ssim tuning
# we plot psnr separately since psnr values are in different ranges compared to ssim and lpips
python3 -u ../plot_curves.py --name livecell_image_metrics_spaghetti_cyclegan \
--type box \
--log_dir \
path_to/translation_stats_livecell_crop_0.5_ssim_lpips.txt \
path_to/translation_stats_livecell_crop_1_ssim_lpips.txt \
path_to/translation_stats_livecell_crop_1.5_ssim_lpips.txt \
path_to/translation_stats_livecell_crop_2_ssim_lpips.txt \
path_to/translation_stats_livecell_crop_3_ssim_lpips.txt \
path_to/translation_stats_livecell_crop_10_ssim_lpips.txt \
path_to/translation_stats_livecell_crop_0.5_ssim_ssim.txt \
path_to/translation_stats_livecell_crop_1_ssim_ssim.txt \
path_to/translation_stats_livecell_crop_1.5_ssim_ssim.txt \
path_to/translation_stats_livecell_crop_2_ssim_ssim.txt \
path_to/translation_stats_livecell_crop_3_ssim_ssim.txt \
path_to/translation_stats_livecell_crop_10_ssim_ssim.txt \
--save_dir path_to_save/ \
--labels 0.5_SSIM_LPIPS 1_SSIM_LPIPS 1.5_SSIM_LPIPS 2_SSIM_LPIPS 3_SSIM_LPIPS 10_SSIM_LPIPS \
0.5_SSIM_SSIM 1_SSIM_SSIM 1.5_SSIM_SSIM 2_SSIM_SSIM 3_SSIM_SSIM 10_SSIM_SSIM

python3 -u ../plot_curves.py --name livecell_image_metrics_spaghetti_cyclegan_psnr \
--type box \
--log_dir \
path_to/translation_stats_livecell_crop_0.5_ssim_psnr.txt \
path_to/translation_stats_livecell_crop_1_ssim_psnr.txt \
path_to/translation_stats_livecell_crop_1.5_ssim_psnr.txt \
path_to/translation_stats_livecell_crop_2_ssim_psnr.txt \
path_to/translation_stats_livecell_crop_3_ssim_psnr.txt \
path_to/translation_stats_livecell_crop_10_ssim_psnr.txt \
--save_dir path_to_save/ \
--labels 0.5_SSIM_PSNR 1_SSIM_PSNR 1.5_SSIM_PSNR 2_SSIM_PSNR 3_SSIM_PSNR 10_SSIM_PSNR

