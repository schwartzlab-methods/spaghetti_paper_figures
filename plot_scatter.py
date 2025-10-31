import os
import numpy as np
import pandas as pd
import altair as alt
import argparse
import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

def correlation(f1, f2):
    data1 = np.load(f1).flatten()
    data2 = np.load(f2).flatten()
    res = pearsonr(data1, data2)
    print(res.statistic, res.pvalue)

def plot_scatter(f1, f2, save, name):
    if f1.endswith(".npy"):
        data = pd.DataFrame({"true":np.load(f1).flatten(), "pred":np.load(f2).flatten()})
        line = alt.Chart(data).mark_point().encode(
            x=alt.X("true:Q", title="True"),
            y=alt.Y("pred:Q", title="Predict"),
        ).interactive()
        correlation(f1, f2)
    else:
        data = pd.read_csv(f1)
        line = alt.Chart(data).mark_point().encode(
            x=alt.X("true:Q", title="True"),
            y=alt.Y("pred:Q", title="Predict"),
        ).interactive()
    # generate a linear regression line
    line = line + alt.Chart(data).mark_line(color="red").transform_regression(
        "true", 
        "pred"
    ).encode(
        x=alt.X("true:Q", title="True"),
        y=alt.Y("pred:Q", title="Predict")
    )
    
    line.save(os.path.join(save, f"{name}_scatter.html"))

    res = pearsonr(data["true"],data["pred"])
    print(res.statistic, res.pvalue)

def main():
    argparser = argparse.ArgumentParser(description="Plotting curves")
    argparser.add_argument("--f1", type=str, help="np1 or csv1")
    argparser.add_argument("--f2", type=str, help="np2, not needed if doing csv")
    argparser.add_argument("--save_dir", type=str, help="The directory to save the plots")
    argparser.add_argument("--name", type=str, help="The name of the experiment")
    args = argparser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    plot_scatter(args.f1, args.f2, args.save_dir, args.name)


if __name__ == "__main__":
    main()