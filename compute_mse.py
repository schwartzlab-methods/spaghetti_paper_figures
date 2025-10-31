'''
compute the mean square error between the inference and the ground truth
'''
from scipy.stats import ttest_rel
import os
import argparse
import numpy as np
import altair as alt
import pandas as pd
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

def main():
    parser = argparse.ArgumentParser(description="Compute the mean square error between the inference and the ground truth")
    parser.add_argument("--inference", type=str, required=True, nargs="+", help="Path(s) to the inference file")
    parser.add_argument("--inference_name", type=str, required=True, nargs="+", help="Name of the inference file")
    parser.add_argument("--ground_truth", type=str, help="Path to the ground truth file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the results")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    # load the inference and the ground truth
    mse_L = []
    label_L = []
    mse_unflatten_L = []
    if args.inference[0].endswith(".npy"):
        ground_truth = np.load(args.ground_truth)
        assert len(args.inference) == len(args.inference_name), "Inference and inference name must have the same length"
        for i, each in enumerate(args.inference):
            inference = np.load(each)
            assert inference.shape == ground_truth.shape, "Inference and ground truth must have the same shape"
            mse = np.square(((inference - ground_truth))).reshape(-1).tolist()
            mse_unflatten_L.append(mse)
            mse_L.extend(mse)
            label_L.extend([args.inference_name[i]]*len(mse))
    else:
        for i, f in enumerate(args.inference):
            df = pd.read_csv(f)
            ground_truth = df["true"].to_numpy()
            inference = df["pred"].to_numpy()
            mse = np.square(((inference - ground_truth))).reshape(-1).tolist()
            mse_unflatten_L.append(mse)
            mse_L.extend(mse)
            label_L.extend([args.inference_name[i]]*len(mse))

    # compute p-values
    print("==========Paried T Test==========")
    for i, item in enumerate(mse_unflatten_L):
        if i%2 == 1:
            continue
        if i == len(mse_unflatten_L)-1:
            continue
        data1 = mse_unflatten_L[i]
        data2= mse_unflatten_L[i+1]
        print(f"Mean of data {i} and {i+1}:", np.mean(data1), np.mean(data2))
        t_stat, p_value = ttest_rel(np.array(data1).flatten(), np.array(data2).flatten())
        print(f"T-statistic: {t_stat}, p-value: {p_value} for rel t_test")
    # construct the dataframe
    data = {
        "Inference": label_L,
        "MSE": mse_L
    }
    df = pd.DataFrame(data)
    # plot the box plot
    plot = alt.Chart(df).mark_boxplot().encode(
        x=alt.X("Inference:N", title="Experiment", sort=args.inference_name, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("MSE:Q", title="Mean Square Error", scale={"type":"log"}),
        color=alt.Color("Inference:N", title="Inference")
    ).interactive()
    # save the plot
    plot.save(os.path.join(args.save_dir, "mse_boxplot.html"))

if __name__ == "__main__":
    main()