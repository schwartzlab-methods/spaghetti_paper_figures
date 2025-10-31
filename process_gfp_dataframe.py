import pandas as pd
import os
import numpy as np
import argparse

def rewrite(column):
    final_L = []
    idx_L = []
    for i, row in enumerate(column):
        row = row.split("\n")
        for x in row:
            x = x.replace("[","").replace("]", "")
            if x != "":
                x = float(x)
                final_L.append(x)
                idx_L.append(i)
    return final_L, idx_L

def main():
    parser = argparse.ArgumentParser(description="Process csv files")
    parser.add_argument("--dir", type=str, required=True, help="Path to the csv directories")
    args = parser.parse_args()
    path = args.dir
    file_L = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            file_L.append(os.path.join(path, file))
    for csv in file_L:
        try:
            f = pd.read_csv(csv)
            true, _ = rewrite(f["true"])
            pred, idx_L = rewrite(f["pred"])
            data = pd.DataFrame({"true": true, "pred": pred, "idx": idx_L})
            data.to_csv(csv)
        except:
            continue

if __name__ == "__main__":
    main()