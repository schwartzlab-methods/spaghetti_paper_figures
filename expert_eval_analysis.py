import os
import altair as alt
import argparse
import pandas as pd
import numpy as np

## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

def module_one_confusion_matrix(eval_df, key_pairs, output_dir):
    confusion_matrix = np.zeros((2,4)) # 4 models - real, SPAGHETTI, CycleGAN, UTOM; 2 labels - real, AI
    # process each row one by one
    for index, row in eval_df.iterrows():
        set_number = row["What is your Set number? This is indicated in the name of the folder for module 1 you received from email."].split(" ")[-1]
        key_pair_path = [key for key in key_pairs if f"set_{set_number}_" in key][0]
        # convert the key pair string to a dictionary stored in txt
        key_pair_dict = {}
        with open(key_pair_path, "r") as f:
            for line in f:
                image_name, label = line.strip().split("\t")
                key_pair_dict[label] = image_name
        for i in range(1, 11):
            user_label = row[f"Please rate each of the images as either Real Acquisition or AI Generated. [{i}.png]"]
            true_label = key_pair_dict[str(i)]
            if "gt" in true_label:
                if user_label == "Real Acquisition":
                    confusion_matrix[0][0] += 1  # True Positive
                else:
                    confusion_matrix[1][0] += 1  # False Negative
            elif "spaghetti" in true_label:
                if user_label == "Real Acquisition":
                    confusion_matrix[0][1] += 1  # False Positive
                else:
                    confusion_matrix[1][1] += 1  # True Negative
            elif "cyclegan" in true_label:
                if user_label == "Real Acquisition":
                    confusion_matrix[0][2] += 1  # False Positive
                else:
                    confusion_matrix[1][2] += 1  # True Negative
            elif "utom" in true_label:
                if user_label == "Real Acquisition":
                    confusion_matrix[0][3] += 1  # False Positive
                else:
                    confusion_matrix[1][3] += 1  # True Negative
    # normalize confusion matrix
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=0, keepdims=True)
    # create a dataframe for plotting
    df_cm = pd.DataFrame(confusion_matrix, columns=["Real", "SPAGHETTI", "CycleGAN", "UTOM"], index=["Real Acquisition", "AI Generated"])
    df_cm = df_cm.reset_index().melt(id_vars="index", var_name="Model", value_name="Percentage")
    # plot confusion matrix using altair
    cm_chart = alt.Chart(df_cm).mark_rect().encode(
        x=alt.X("Model:N", title="Model"),
        y=alt.Y("index:N", title="User Label"),
        color=alt.Color("Percentage:Q", scale=alt.Scale(scheme="blues"), title="Percentage"),
        tooltip=[alt.Tooltip("Model:N", title="Model"), alt.Tooltip("index:N", title="User Label"), alt.Tooltip("Percentage:Q", title="Percentage", format=".2f")]
    ).properties(
        title="Confusion Matrix for Module 1"
    ).interactive()
    cm_chart.save(os.path.join(output_dir, "module_1_confusion_matrix.html"))


def module_two_box_plots(eval_df, key_pairs, output_dir):
    ranks_dict = {"SPAGHETTI": [], "CycleGAN": [], "UTOM": [], "Original_UTOM": []}
    quality_dict = {"SPAGHETTI": [], "CycleGAN": [], "UTOM": [], "Original_UTOM": []}
    # process each row one by one
    for index, row in eval_df.iterrows():
        for i in range(1, 11): # iterating through 10 different groups
            key_pair_path = [key for key in key_pairs if f"group_{i}.txt" in key][0]
            with open(key_pair_path, "r") as f:
                key_pair_dict = {}
                for line in f:
                    image_name, label = line.strip().split("\t")
                    key_pair_dict[label] = image_name
            # process each image
            for img in range(1,5):
                if img == 4:
                    imge_type = "original_utom_img"
                else:
                    imge_type = key_pair_dict[f"{img}.png"]
                # compute rank for each image
                user_rank = [x for x in str(row[f"group {i}: Please rank the images based on biological realism by comparing that to the original PCM. Please do not rank duplicate. [{img}.png]"]) if x.isdigit()][0]
                # compute quality for each image
                user_quality = [x for x in str(row[f"group {i}: Please rank {img}.png based on its image quality"]) if x.isdigit()][0]
                # add to dict
                if "spaghetti" in imge_type:
                    ranks_dict["SPAGHETTI"].append(int(user_rank))
                    quality_dict["SPAGHETTI"].append(int(user_quality))
                elif "cyclegan" in imge_type:
                    ranks_dict["CycleGAN"].append(int(user_rank))
                    quality_dict["CycleGAN"].append(int(user_quality))
                elif "original_utom" in imge_type:
                    ranks_dict["Original_UTOM"].append(int(user_rank))
                    quality_dict["Original_UTOM"].append(int(user_quality))
                elif "utom" in imge_type:
                    ranks_dict["UTOM"].append(int(user_rank))
                    quality_dict["UTOM"].append(int(user_quality))
    # create dataframes for plotting
    df_ranks = pd.DataFrame({k: pd.Series(v) for k, v in ranks_dict.items()}).melt(var_name="Model", value_name="Rank")
    df_quality = pd.DataFrame({k: pd.Series(v) for k, v in quality_dict.items()}).melt(var_name="Model", value_name="Quality")
    # plot ranks
    rank_chart = alt.Chart(df_ranks).mark_boxplot().encode(
        x=alt.X("Model:N", title="Model"),
        y=alt.Y("Rank:Q", title="Rank"),
        color=alt.Color("Model:N", title="Model"),
        tooltip=[alt.Tooltip("Model:N", title="Model"), alt.Tooltip("Rank:Q", title="Rank")]
    ).properties(
        title="Box Plot of Ranks for Module 2"
    ).interactive()
    rank_chart.save(os.path.join(output_dir, "module_2_ranks_boxplot.html"))
    # plot quality
    quality_chart = alt.Chart(df_quality).mark_boxplot().encode(
        x=alt.X("Model:N", title="Model"),
        y=alt.Y("Quality:Q", title="Quality"),
        color=alt.Color("Model:N", title="Model"),
        tooltip=[alt.Tooltip("Model:N", title="Model"), alt.Tooltip("Quality:Q", title="Quality")]
    ).properties(
        title="Box Plot of Image Quality for Module 2"
    ).interactive()
    quality_chart.save(os.path.join(output_dir, "module_2_quality_boxplot.html"))

def main():
    parser = argparse.ArgumentParser(description="Expert evaluation of spaghetti figures")
    parser.add_argument("--input_tsv", type=str, nargs="+", required=True, help="Path to the input TSV file containing evaluation data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output plots")
    parser.add_argument("--dict_dir", type=str, help="Path to all image_name and their corresponding labels in the format image_name:label")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the evaluation data
    dfs = []
    for input_tsv in args.input_tsv:
        dfs.append(pd.read_csv(input_tsv, sep="\t", header=0))
    eval_df = pd.concat(dfs, ignore_index=True)

    all_dict = [os.path.join(args.dict_dir, x) for x in os.listdir(args.dict_dir) if x.endswith(".txt")]

    # generate confusion matrix for module 1
    module_one_confusion_matrix(eval_df, all_dict, args.output_dir)

    # generate box plots for image quality and ranking for module 2
    module_two_box_plots(eval_df, all_dict, args.output_dir)

if __name__ == "__main__":
    main()