'''
Plot the AUC vs. dataset size for the different models
Colour each point by the model and the dataset
'''

import altair as alt
import pandas as pd
import numpy as np
import os
import argparse
from bs4 import BeautifulSoup
import json
import re

## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

# dictionary of model names and sizes
model_training_sizes = {"SPAGHETTI": 1.3*(10**3),
                        "UNI": 100*(10**6),
                        "Phikon": 456*(10**6),}

def generate_panda_data(data_path):
    '''
    Generate a pandas dataframe from the data
    '''
    with open(data_path, "r") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    script_tag = soup.find("script", text=re.compile(r"var spec"))
    script_content = script_tag.string

    # Use a regular expression to extract the JSON object from "var spec"
    match = re.search(r"var spec = (\{.*?\});", script_content, re.DOTALL)
    if match:
        spec_json = match.group(1)  # The JSON object as a string
        spec = json.loads(spec_json)  # Parse it into a Python dictionary

        # Extract the dataset
        dataset = list(spec.get("datasets", {}).values())[0]
    else:
        raise ValueError("No match found")
    return dataset

def process_dataset(dataset):
    '''
    Compute the mean of AUC and the std values for each model combination
    '''
    temp_data = {}
    for each in dataset:
        exp_name = each["Experiment"]
        if (("cyclegan" in exp_name.lower()) or ("vit" in exp_name.lower()) or ("resnet" in exp_name.lower()) 
             or ("utom" in exp_name.lower()) or ("optimus" in exp_name.lower())):
            continue
        if exp_name not in temp_data:
            temp_data[exp_name] = [each["AUC"]]
        else:
            temp_data[exp_name].append(each["AUC"])
    final_data = []
    for key, value in temp_data.items():
        final_data.append({"experiment": key, "auc": np.mean(value), "std": np.std(value), 'exp_type': key.split("+")[-1]})
    return final_data

def calculate_data_size(name):
    if "+" in name:
        names = name.split("+")
        size = 0
        for each in names:
            size += model_training_sizes[each]
        return size
    else:
        return model_training_sizes[name]

def plot_auc_vs_dataset_size(data,output):
    df = pd.DataFrame(data)
    df["size"] = df["experiment"].apply(calculate_data_size)
    # plot if Phikon is in the name, use circle, else use triangle
    plot = alt.Chart(df).mark_point().encode(
        x=alt.X("size:Q", scale=alt.Scale(type="log"), title="Number of Training Slides"),
        y=alt.Y("auc:Q", title="AUC", scale=alt.Scale(domain=[0.9,0.97])),
        shape="exp_type:N",
        # shape=shape_condition,
            #alt.expr.indexof(alt.datum.experiment, "UNI") > -1, 
        #title="Model")
    )
    # plot = alt.Chart(df).mark_point().encode(
    #     x=alt.X("size:Q", scale=alt.Scale(type="log"), title="Dataset Size"),
    #     y=alt.Y("auc:Q", title="AUC", scale=alt.Scale(domain=[0.7,1])),
    #     color=alt.Color("experiment:N", title="Model"),
    #     tooltip=["experiment", "auc", "std"]
    # )
    # plot = plot + alt.Chart(df).mark_errorbar().encode(
    #     x=alt.X("size:Q", scale=alt.Scale(type="log")),
    #     y=alt.Y("auc:Q", title="AUC", scale=alt.Scale(domain=[0.7,1])),
    #     yError="std:Q"
    # )
    name = os.path.join(output, f"auc_vs_datasetsize.html")
    plot.interactive().save(name)

def main():
    dataset = generate_panda_data(args.data)
    data = process_dataset(dataset)
    plot_auc_vs_dataset_size(data,args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot AUC vs. dataset size for different models")
    parser.add_argument("--data", type=str, help="Path to the Altair HTML file of the box plot")
    parser.add_argument("--output", type=str, help="Path to the output file")
    args = parser.parse_args()
    main()
