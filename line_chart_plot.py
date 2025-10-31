import pandas as pd
import altair as alt
import argparse
import os
import numpy as np

## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

def plot_line_chart_no_std(data: pd.DataFrame, x_title: str, y_title: str, 
                    save: str, name: str="line_chat_no_std"):
    '''
    Create a line chart
    '''
    line = alt.Chart(data).mark_line().encode(
        x=alt.X("x:Q", title=x_title),
        y=alt.Y("y_mean:Q", title=y_title),
        color=alt.Color("exp_name:N", title="Experiment Name"),
        strokeDash=alt.StrokeDash("exp_type:N", title="Experiment Type")
    ).interactive()
    line.save(os.path.join(save, f"{name}.html"))

def plot_line_chart_with_std(data: pd.DataFrame, x_title: str, y_title: str, 
                            save: str, name: str="noise_addition_std"):
    '''
    Create a line chart with standard deviation shading
    '''
    line = alt.Chart(data).mark_line().encode(
        x=alt.X("x:Q", title=x_title),
        y=alt.Y("y_mean:Q", title=y_title),
        color=alt.Color("exp_name:N", title="Experiment Name"),
        strokeDash=alt.StrokeDash("exp_type:N", title="Experiment Type")
    ).interactive()
    band = alt.Chart(data).mark_area(opacity=0.3).encode(
        x=alt.X("x:Q"),
        y=alt.Y("y_max:Q", title=y_title),
        y2=alt.Y2("y_min:Q"),
        color=alt.Color("exp_name:N", title="Experiment Name"),
        strokeDash=alt.StrokeDash("exp_type:N", title="Experiment Type")
    )
    # band = alt.Chart(data).mark_errorbar().encode(
    #     x='x:Q',
    #     y=alt.Y('y_min:Q', title=None),
    #     y2='y_max:Q',
    #     color='exp_name:N'
    # )
    (band + line).interactive().save(os.path.join(save, f"{name}_with_std.html"))

def get_data_no_std(data_x_L: list[str], data_y_L: list[str], 
                    exp_names_L: list[str], exp_types_L: list[str]) -> pd.DataFrame:
    x_L = []
    y_L = []
    exp_name_L = []
    exp_type_L = []
    for each in data_x_L:
        with open(each, "r") as f:
            data_x = f.readlines()
            data_x = [float(x.strip()) for x in data_x]
            x_L.extend(data_x)
    for i, each in enumerate(data_y_L):
        with open(each, "r") as f:
            data_y = f.readlines()
            data_y = [float(y.strip()) for y in data_y]
            exp_names = [exp_names_L[i] for _ in range(len(data_y))]
            exp_types = [exp_types_L[i] for _ in range(len(data_y))]
            y_L.extend(data_y)
            exp_name_L.extend(exp_names)
            exp_type_L.extend(exp_types)
    pd_data = pd.DataFrame({"x": x_L, "y": y_L, "exp_name": exp_name_L, "exp_type": exp_type_L})
    return pd_data

def get_data_with_std(data_x_L: list[str], data_y_L: list[str], 
                      exp_names_L: list[str], exp_types_L: list[str]) -> pd.DataFrame:
    num_files_per_exp = len(data_y_L) // len(exp_names_L)
    print(f"Number of files per experiment: {num_files_per_exp}")
    x_L = []
    y_mean_L = []
    y_std_L = []
    exp_name_L = []
    exp_type_L = []

    y_temp_mean = []
    y_temp_std = []
    for i, each in enumerate(data_y_L):
        if (i+1) % num_files_per_exp == 0:
            with open(data_x_L[i // num_files_per_exp], "r") as f:
                data_x = f.readlines()
                data_x = [float(x.strip()) for x in data_x]
                x_L.extend(data_x)
            if num_files_per_exp == 1:
                y_std_L.extend([0.0 for _ in range(len(data_x))])
                with open(data_y_L[i], "r") as f:
                    data_y = f.readlines()
                    data_y = [float(y.strip()) for y in data_y]
                    y_mean_L.extend(data_y)
            else:
                y_std_L.extend(y_temp_std)
                y_mean_L.extend(y_temp_mean)
            exp_names = [exp_names_L[i // num_files_per_exp] for _ in range(len(data_x))]
            exp_types = [exp_types_L[i // num_files_per_exp] for _ in range(len(data_x))]
            exp_name_L.extend(exp_names)
            exp_type_L.extend(exp_types)
            y_temp_mean = []
            y_temp_std = []
        else:
            with open(data_y_L[i], "r") as f:
                data_y = f.readlines()
                data_y = [float(y.strip()) for y in data_y]
                y_temp_mean.append(np.mean(data_y))
                y_temp_std.append(np.std(data_y))
    print(f"Length of x_L: {len(x_L)}, y_mean_L: {len(y_mean_L)}, y_std_L: {len(y_std_L)}, exp_name_L: {len(exp_name_L)}, exp_type_L: {len(exp_type_L)})")
    pd_data = pd.DataFrame({"x": x_L, "y_mean": y_mean_L, 
                            "y_max": [y_m + y_s for y_m, y_s in zip(y_mean_L, y_std_L)],
                            "y_min": [y_m - y_s for y_m, y_s in zip(y_mean_L, y_std_L)],
                            "exp_name": exp_name_L, "exp_type": exp_type_L})
    return pd_data

def main(data_x, data_y, exp_names, exp_types, x_title, y_title, save, name, with_std):
    pd_data = get_data_with_std(data_x, data_y, exp_names, exp_types)
    if with_std:
        plot_line_chart_with_std(pd_data, x_title, y_title, save, name)
    else:
        # pd_data = get_data_no_std(data_x, data_y, exp_names, exp_types)
        plot_line_chart_no_std(pd_data, x_title, y_title, save)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot line chart")
    parser.add_argument("--data_x", type=str, nargs="+", help="Data for x axis")
    parser.add_argument("--data_y", type=str, nargs="+", help="Data for y axis")
    parser.add_argument("--exp_names", type=str, nargs="+", help="Experiment names (ie: feature extractor + dataset)")
    parser.add_argument("--exp_types", type=str, nargs="+", help="Experiment types (ie: the model)")
    parser.add_argument("--x_title", type=str, help="Title of the x axis")
    parser.add_argument("--y_title", type=str, help="Title of the y axis")
    parser.add_argument("--save", type=str, help="Save location")
    parser.add_argument("--name", type=str, default="line_chart", help="Name of the plot")
    parser.add_argument("--with_std", action="store_true", help="Whether to plot with standard deviation shading")
    args = parser.parse_args()
    os.makedirs(args.save, exist_ok=True)
    main(args.data_x, args.data_y, args.exp_names, args.exp_types, 
         args.x_title, args.y_title, args.save, args.name, args.with_std)

