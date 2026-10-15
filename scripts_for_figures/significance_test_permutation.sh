# significance test, repeat for any two groups of your liking
# this particular test reads the AUC values from a generated json file
python3 -u ../stats_test.py \
--f1 path_to_exp_1_txt_file_with_dice_scores \
--f2 path_to_exp_2_txt_file_with_dice_scores \
--test permutation