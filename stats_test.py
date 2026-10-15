from scipy.stats import ttest_rel, rankdata, mannwhitneyu, wilcoxon
import json
import argparse
import numpy as np
from tqdm import tqdm

def get_list_micro(data):
    res = []
    for each in data.values():
        res.append(each["roc_auc"]["micro"])
    return res

def permutation_test(data1, data2, n_permutations = 100000):
    # convert lists (if lists) to np arrays
    data1 = np.array(data1).reshape(-1)
    data2 = np.array(data2).reshape(-1)
    
    # Observed statistic (mean difference)
    observed_diff = np.mean(data1) - np.mean(data2)

    # Combine the AUC values
    total = np.concatenate([data1,data2]).flatten()

    # Initialize list to store permuted statistics
    permuted_diffs = []

    # Permutation loop
    rng = np.random.default_rng()
    # for _ in tqdm(range(n_permutations)):
    for _ in range(n_permutations):
        combined = rng.permuted(total)
        # np.random.shuffle(combined)  # Shuffle the combined values
        permuted_a = combined[:len(data1)]  # Split into two groups
        permuted_b = combined[len(data2):]
        permuted_diff = np.mean(permuted_a) - np.mean(permuted_b)
        permuted_diffs.append(permuted_diff)

    # Compute the p-value
    permuted_diffs = np.array(permuted_diffs)
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))

    # Results
    print(f"Result for two-tailed permutation test of {n_permutations} permutations:")
    print(f"Observed difference: {observed_diff}")
    print(f"p-value: {p_value}")
    print("Sample sizes:", len(data1), len(data2))
    print("Effect Size (Cohen's d):", (np.mean(data1) - np.mean(data2)) / np.sqrt((np.std(data1) ** 2 + np.std(data2) ** 2) / 2))

def mul_rank_product(data1, data2):
    auc_values = np.stack([data1, data2], axis=2)  # Shape: (datasets, folds, classifiers)

    # Step 1: Compute ranks within each dataset and fold
    ranks = np.empty_like(auc_values)
    for i in range(auc_values.shape[0]):  # Loop over datasets
        for j in range(auc_values.shape[1]):  # Loop over folds
            ranks[i, j, :] = rankdata(auc_values[i, j, :])  # Rank classifiers in each fold

    # Step 2: Compute the rank product for each classifier
    rank_products = np.prod(ranks, axis=(0, 1)) ** (1 / (auc_values.shape[0] * auc_values.shape[1]))

    # Step 3: Permutation test to compute p-values
    n_datasets, n_folds, n_classifiers = auc_values.shape
    n_permutations = 10000
    permuted_rank_products = []

    for _ in range(n_permutations):
        permuted_ranks = np.empty_like(ranks)
        for i in range(n_datasets):
            for j in range(n_folds):
                permuted_ranks[i, j, :] = rankdata(np.random.permutation(auc_values[i, j, :]))
        permuted_rps = np.prod(permuted_ranks, axis=(0, 1)) ** (1 / (n_datasets * n_folds))
        permuted_rank_products.append(permuted_rps)

    permuted_rank_products = np.array(permuted_rank_products)

    # Compute p-values for each classifier
    p_values = [(permuted_rank_products[:, c] <= rank_products[c]).mean() for c in range(n_classifiers)]

    # Output results
    print(f"Rank Products: {rank_products}")
    print(f"P-values: {p_values}")

def rank_product_test(data1, data2):
    # Combine into a single matrix
    auc_values = np.array([data1, data2]).T  # Shape: (datasets, classifiers)

    # Step 1: Compute ranks within each dataset
    ranks = np.apply_along_axis(rankdata, 1, auc_values)  # Rank within each dataset

    # Step 2: Compute rank product for each classifier
    rank_products = np.prod(ranks, axis=0) ** (1 / ranks.shape[0])

    # Step 3: Permutation test to compute p-values
    n_datasets, n_classifiers = ranks.shape
    n_permutations = 10000
    permuted_rank_products = []

    for _ in range(n_permutations):
        permuted_ranks = np.apply_along_axis(lambda x: rankdata(np.random.permutation(x)), 1, auc_values)
        permuted_rps = np.prod(permuted_ranks, axis=0) ** (1 / n_datasets)
        permuted_rank_products.append(permuted_rps)

    permuted_rank_products = np.array(permuted_rank_products)

    # Compute p-values
    p_values = [(permuted_rank_products[:, j] <= rank_products[j]).mean() for j in range(n_classifiers)]

    # Output results
    print(f"Rank Products: {rank_products}")
    print(f"P-values: {p_values}")


def main(json_1, json_2, if_txt, if_npy, test):
    np.random.seed(42)  # For reproducibility
    if args.npy:
        data_1 = np.load(json_1[0]).flatten()
        data_2 = np.load(json_2[0]).flatten()
    else:
        data_1 = []
        data_2 = []
        if not if_txt:
            for each in json_1:
                with open(each, "r") as f:
                    micro_1 = json.load(f)
                    micro_1 = get_list_micro(micro_1)
                data_1.append(micro_1)
            for each in json_2:
                with open(each, "r") as f:
                    micro_2 = json.load(f)
                    micro_2 = get_list_micro(micro_2)
                data_2.append(micro_2)
        else:
            for each in json_1:
                with open(each, "r") as f:
                    data_1_temp = f.readlines()
                    data_1_temp = [float(line.strip()) for line in data_1_temp]
                data_1.append(data_1_temp)
            for each in json_2:
                with open(each, "r") as f:
                    data_2_temp = f.readlines()
                    data_2_temp = [float(line.strip()) for line in data_2_temp]
                data_2.append(data_2_temp)
    if test == "paired_t_test":
        print("==========Paried T Test==========")
        print("Mean of data 1 and 2:", np.mean(data_1), np.mean(data_2))
        t_stat, p_value = ttest_rel(np.array(data_1).flatten(), np.array(data_2).flatten())
        print(f"T-statistic: {t_stat}, p-value: {p_value} for rel t_test")
    if test == "mann_whitney":
        print("==========Mann-Whitney U Test==========")
        print("Mean of data 1 and 2:", np.mean(data_1), np.mean(data_2))
        u_stat, p_value = mannwhitneyu(np.array(data_1).flatten(), np.array(data_2).flatten())
        print(f"U-statistic: {u_stat}, p-value: {p_value} for Mann-Whitney U test")
        print("Sample sizes:", len(np.array(data_1).flatten()), len(np.array(data_2).flatten()))
        print("Effect Size using rank-biserial correlation:", 1 - (2 * u_stat) / (len(np.array(data_1).flatten()) * len(np.array(data_2).flatten())))
    if test == "wilcoxon":
        print("==========Wilcoxon Signed-Rank Test==========")
        print("Mean of data 1 and 2:", np.mean(data_1), np.mean(data_2))
        w_stat, p_value = wilcoxon(np.array(data_1).flatten(), np.array(data_2).flatten())
        print(f"W-statistic: {w_stat}, p-value: {p_value} for Wilcoxon signed-rank test")
        print("Sample sizes:", len(np.array(data_1).flatten()), len(np.array(data_2).flatten()))
        # Effect size calculation for Wilcoxon
        n = len(np.array(data_1).flatten())
        z = (w_stat - (n * (n + 1)) / 4) / np.sqrt((n * (n + 1) * (2 * n + 1)) / 24)
        print("Effect Size (r):", np.abs(z) / np.sqrt(n))
    if test == "rank_product":
        print("==========Ranked Product Test==========")
        rank_product_test(data_1, data_2)
    if test == "mul_rank_product":
        print("==========Multi Ranked Product Test==========")
        mul_rank_product(data_1, data_2)
    if test == "permutation":
        print("==========Permutation Test==========")
        print("Mean of data 1 and 2:", np.mean(data_1), np.mean(data_2))
        permutation_test(np.array(data_1).flatten(), np.array(data_2).flatten())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f1', nargs="+", type=str, help='Path(s) to file 1')
    parser.add_argument('--f2', nargs="+", type=str, help='Path(s) to file 2')
    parser.add_argument('--txt', action="store_true", help='Read txt instead of json file')
    parser.add_argument('--npy', action="store_true", help='Read npy instead of json file')
    parser.add_argument('--test', type=str, help="The statistics test to be used")
    args = parser.parse_args()
    main(args.f1, args.f2, args.txt, args.npy, args.test)