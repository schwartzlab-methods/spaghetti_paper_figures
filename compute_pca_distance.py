'''
Compute the PCA-based distance between cell lines and tissues using a precomputed distance matrix
'''
from tqdm import tqdm
from scipy.stats import mannwhitneyu
import os
import numpy as np
import pandas as pd
import argparse
import altair as alt
## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

def box_plot(matrix1, matrix2, save_path):
    '''Create box plots comparing the distances in two matrices of diagonal elements vs off-diagonal elements

    Args:
        matrix1 (pd.DataFrame): The first distance matrix
        matrix2 (pd.DataFrame): The second distance matrix
        save_path (str): The path to save the box plot
    '''
    diag_elements_1 = []
    off_diag_elements_1 = []
    diag_elements_2 = []
    off_diag_elements_2 = []
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            if i == j:
                diag_elements_1.append(matrix1.iloc[i, j])
                diag_elements_2.append(matrix2.iloc[i, j])
            else:
                off_diag_elements_1.append(matrix1.iloc[i, j])
                off_diag_elements_2.append(matrix2.iloc[i, j])
    boxplot_data = pd.DataFrame({
        "Distance": diag_elements_1 + off_diag_elements_1 + diag_elements_2 + off_diag_elements_2,
        "Type": ["Diagonal"] * len(diag_elements_1) + ["Off-Diagonal"] * len(off_diag_elements_1) +
                ["Diagonal"] * len(diag_elements_2) + ["Off-Diagonal"] * len(off_diag_elements_2),
        "Matrix": ["Matrix 1_Diagonal"] * len(diag_elements_1) + ["Matrix 1_Off-Diagonal"] * len(off_diag_elements_1) +
                  ["Matrix 2_Diagonal"] * len(diag_elements_2) + ["Matrix 2_Off-Diagonal"] * len(off_diag_elements_2)
    })
    boxplot = alt.Chart(boxplot_data).mark_boxplot().encode(
        x=alt.X('Matrix:N', title='Element Type', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Distance:Q', title='Distance'),
        color='Type:N'
    )
    boxplot.interactive().save(os.path.join(save_path, "distance_boxplot.html"))

def compute_pca_distance(matrix, list_of_cell_lines, dict, save_path):
    '''Restructure the matrix to be num_cell_lines x num_tissue_samples

    Args:
        matrix (pd.DataFrame): The precomputed distance matrix of shape (num_cell_lines + num_tissue_samples) x (num_cell_lines + num_tissue_samples)
        list_of_cell_lines (list): The list of cell line names
        dict (dict): A dictionary that relates the tissue to the cell line of the same origin, where keys are tissue names and values are cell line names
        save_path (str): The path to save the PCA distance matrix
    '''
    # Extract the indices for cell lines and tissues
    cell_line_indices = [matrix.index.get_loc(cell_line) for cell_line in list_of_cell_lines]
    tissue_indices = [matrix.index.get_loc(tissue) for tissue in dict.keys()]

    # Subset the matrix
    pca_matrix = matrix.iloc[cell_line_indices, tissue_indices]

    # collpase cell lines of the same tissue origin by taking the mean
    collapsed_rows = []
    collapsed_row_names = []
    for tissue, cell_lines in dict.items():
        cell_line_rows = pca_matrix.loc[cell_lines]
        collapsed_row = cell_line_rows.mean(axis=0)
        collapsed_rows.append(collapsed_row)
        collapsed_row_names.append(tissue)
    pca_matrix = pd.DataFrame(collapsed_rows, index=collapsed_row_names, columns=pca_matrix.columns)

    return pca_matrix

def compute_distance_all_features(matrix, dict):
    '''
    Get each distance and classify them as within-tissue or cross-tissue
    '''
    within_tissue_distances = []
    cross_tissue_distances = []
    for i in tqdm(range(matrix.shape[0])):
        for j in range(matrix.shape[1]):
            cell_line_name = matrix.index[i]
            tissue_name = matrix.columns[j]
            if tissue_name in dict.keys() and cell_line_name in dict[tissue_name]:
                within_tissue_distances.append(matrix.iloc[i, j])
            else:
                cross_tissue_distances.append(matrix.iloc[i, j])
    return within_tissue_distances, cross_tissue_distances


def subset_to_cell_lines(matrix, labels=None):
    '''
    Make this matrix cell lines x tissue. Cell lines have a number in it
    '''
    if labels is None:
        labels = matrix.index
    cell_line_indices = [i for i, name in enumerate(labels) if any(char.isdigit() for char in name)]
    tissue_indices = [i for i, name in enumerate(labels) if not any(char.isdigit() for char in name)]
    subset_matrix = matrix.iloc[cell_line_indices, tissue_indices]
    if labels is not None:
        subset_matrix.index = [labels[i] for i in cell_line_indices]
        subset_matrix.columns = [labels[i] for i in tissue_indices]
    return subset_matrix #shape of num_cell_line_Samples x num_tissues_Samples

def perm_test_centroid_pair(D1, D2, big_D1, big_D2, dict, n_perm=5000, two_sided=False, seed=None):
    '''
    Perform permutation test to assess the significance of the difference in PCA distances between two distance matrices.
    Args:
        D1 (pd.DataFrame): PCA distance matrix 1
        D2 (pd.DataFrame): PCA distance matrix 2
        big_D1 (pd.DataFrame): Original distance matrix 1
        big_D2 (pd.DataFrame): Original distance matrix 2
        dict (dict): A dictionary that relates the tissue to the cell line of the same origin, where keys are tissue names and values are cell line names
        n_perm (int): Number of permutations
        two_sided (bool): Whether to perform a two-sided test
        seed (int): Random seed for reproducibility
    Returns:
        p_dict (dict): Dictionary of p-values for each (A, B) pair
        obs_dict (dict): Dictionary of observed statistics for each (A, B) pair
        perm_stats_dict (dict): Dictionary of permutation statistics for each (A, B) pair
    '''
    rng = np.random.default_rng(seed)
    p_dict = {}
    obs_dict = {}
    perm_stats_dict = {}
    for A in dict.keys():
        # for B in dict[A]:
        obs = D1.loc[A, A] - D2.loc[A, A]
        n = big_D1.shape[0]
        m = big_D2.shape[1]
        perms = rng.permuted(np.arange(n))
        perm_stats = []
        # we permute labels of D2 (full permutation each repeat)
        for _ in range(n_perm):
            A_permuted = rng.permutation(n)
            B_permuted = rng.permutation(m)
            perm_stats.append(big_D1.iloc[A_permuted, B_permuted] - big_D2.iloc[A_permuted, B_permuted])
        perm_stats = np.array(perm_stats)
        if two_sided:
            p = np.mean(np.abs(perm_stats) >= abs(obs))
        else:
            # test D1 < D2  -> obs < 0 is evidence; p = proportion <= obs
            p = np.mean(perm_stats <= obs)
        p_dict[A] = p
        obs_dict[A] = obs
        perm_stats_dict[A] = perm_stats
    return p_dict, obs_dict, perm_stats_dict

def main():
    parser = argparse.ArgumentParser(description="Compute PCA-based distance between cell lines and tissues and assess significance")
    parser.add_argument("--pca_distance_matrix_path", type=str, nargs="+", default=None, help="Paths to the precomputed distance matrix CSV file")
    parser.add_argument("--features_distance_matrix_path", type=str, default=None, nargs="+", help="Path to the feature-wise distance matrix (if applicable)")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the PCA distance matrix")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Load the list of cell lines
    cell_lines_list = ["BT474", "MCF7", "SkBr3", "Huh7", "SKOV3"]

    tissue_cell_line_dict = {
        "Breast": ["BT474", "MCF7", "SkBr3"],
        "Liver": ["Huh7"],
        "Ovarian": ["SKOV3"]
    }

    if args.features_distance_matrix_path:
        distance_matrix_1 = pd.DataFrame(np.load(args.features_distance_matrix_path[0])['distances'])
        distance_matrix_2 = pd.DataFrame(np.load(args.features_distance_matrix_path[1])['distances'])
        distance_matrix_1 = (distance_matrix_1 - distance_matrix_1.min().min()) / (distance_matrix_1.max().max() - distance_matrix_1.min().min())
        distance_matrix_2 = (distance_matrix_2 - distance_matrix_2.min().min()) / (distance_matrix_2.max().max() - distance_matrix_2.min().min())
        labels_1 = np.load(args.features_distance_matrix_path[0])['labels']
        labels_2 = np.load(args.features_distance_matrix_path[1])['labels']
        distance_matrix_1 = subset_to_cell_lines(distance_matrix_1, labels=labels_1)
        print("Distance matrix 1 shape after subsetting:", distance_matrix_1.shape)
        distance_matrix_2 = subset_to_cell_lines(distance_matrix_2, labels=labels_2)
        print("Distance matrix 2 shape after subsetting:", distance_matrix_2.shape)
        within_1, cross_1 = compute_distance_all_features(distance_matrix_1, tissue_cell_line_dict)
        within_2, cross_2 = compute_distance_all_features(distance_matrix_2, tissue_cell_line_dict)
        # create boxplot
        boxplot_data = pd.DataFrame({
            "Distance": within_1 + cross_1 + within_2 + cross_2,
            "Type": ["Within-Tissue"] * len(within_1) + ["Cross-Tissue"] * len(cross_1) +
                    ["Within-Tissue"] * len(within_2) + ["Cross-Tissue"] * len(cross_2),
            "Matrix": ["Matrix 1_Within"] * len(within_1) + ["Matrix 1_Cross"] * len(cross_1) +
                      ["Matrix 2_Within"] * len(within_2) + ["Matrix 2_Cross"] * len(cross_2)
        })

        print("Mean within tissue distance matrix 1:", np.mean(within_1))
        print("Number of within tissue distances matrix 1:", len(within_1))
        print("Mean cross tissue distance matrix 1:", np.mean(cross_1))
        print("Number of cross tissue distances matrix 1:", len(cross_1))
        print("Mean within tissue distance matrix 2:", np.mean(within_2))
        print("Number of within tissue distances matrix 2:", len(within_2))
        print("Mean cross tissue distance matrix 2:", np.mean(cross_2))
        print("Number of cross tissue distances matrix 2:", len(cross_2))

        # run mannwhitneyu test for significance
        print("==============================Test for Matrix 1")
        u_stat, p_value = mannwhitneyu(np.array(within_1).flatten(), np.array(cross_1).flatten())
        print(f"U-statistic: {u_stat}, p-value: {p_value} for Mann-Witney U")
        print("Effect Size using rank-biserial correlation:", 1 - (2 * u_stat) / (len(np.array(within_1).flatten()) * len(np.array(cross_1).flatten())))

        print("==============================Test for Matrix 2")
        u_stat, p_value = mannwhitneyu(np.array(within_2).flatten(), np.array(cross_2).flatten())
        print(f"U-statistic: {u_stat}, p-value: {p_value} for Mann-Witney U")
        print("Effect Size using rank-biserial correlation:", 1 - (2 * u_stat) / (len(np.array(within_2).flatten()) * len(np.array(cross_2).flatten())))

        
        # pre-compute stats for boxplot otherwise there are too many points to be embedded

        box_stats = (boxplot_data.groupby(['Matrix', 'Type'])['Distance']
                     .describe(percentiles=[0.25, 0.5, 0.75])[['min', '25%', '50%', '75%', 'max']]
                     .rename(columns={'25%': 'q1', '50%': 'median', '75%': 'q3'})
                     .reset_index()
        )
        box_stats.to_csv(os.path.join(args.save_dir, "feature_distance_boxplot_stats.csv"), index=False)

        base = alt.Chart(box_stats).encode(
            y=alt.Y('Matrix:N', title='Experiment'),
            yOffset='Type:N',
            color=alt.Color('Type:N', title='Type')
        )

        # Draw boxes (Q1–Q3)
        boxes = base.mark_bar(size=20, opacity=0.8).encode(
            x='q1:Q',
            x2='q3:Q'
        )

        # Median lines
        median = base.mark_rule(size=2, color='black').encode(
            x='median:Q'
        )

        # Whiskers (min and max)
        whiskers = base.mark_rule(color='black').encode(
            x='min:Q',
            x2='max:Q'
        )

        boxplot = (boxes + whiskers + median).configure_scale(
            bandPaddingInner=0.3,  # spacing between type groups
            bandPaddingOuter=0.1
        ).interactive()

        # boxplot = alt.Chart(boxplot_data).mark_boxplot().encode(
        #     x=alt.X('Matrix:N', title='Element Type', axis=alt.Axis(labelAngle=-45)),
        #     y=alt.Y('Distance:Q', title='Distance'),
        #     color='Type:N'
        # ).interactive()
        # boxplot.save(os.path.join(args.save_dir, "feature_distance_boxplot.svg"))


        boxplot.save(os.path.join(args.save_dir, "feature_distance_boxplot.html"))
        # save the features
        np.savez_compressed(os.path.join(args.save_dir, "feature_distance_within_cross_tissue.npz"),
                            within_1=np.array(within_1), cross_1=np.array(cross_1),
                            within_2=np.array(within_2), cross_2=np.array(cross_2))
        print("Feature-wise distance boxplot computed and saved.")
        return 0

    # Load the distance matrix
    distance_matrix_1 = pd.read_csv(args.distance_matrix_path[0], index_col=0)
    distance_matrix_2 = pd.read_csv(args.distance_matrix_path[1], index_col=0)

    # normalize the distance matrix
    # max_both = max(distance_matrix_1.max().max(), distance_matrix_2.max().max())
    # distance_matrix_1 = distance_matrix_1 / max_both
    # distance_matrix_2 = distance_matrix_2 / max_both
    # min_both = min(distance_matrix_1.min().min(), distance_matrix_2.min().min())
    distance_matrix_1 = (distance_matrix_1 - distance_matrix_1.min().min()) / (distance_matrix_1.max().max() - distance_matrix_1.min().min())
    distance_matrix_2 = (distance_matrix_2 - distance_matrix_2.min().min()) / (distance_matrix_2.max().max() - distance_matrix_2.min().min())

    # subset to cell lines x tissues
    distance_matrix_1_sub = subset_to_cell_lines(distance_matrix_1)
    distance_matrix_2_sub = subset_to_cell_lines(distance_matrix_2)


    # Compute PCA distance matrix
    pca_distance_matrix_1 = compute_pca_distance(distance_matrix_1, cell_lines_list, tissue_cell_line_dict, args.save_dir)
    pca_distance_matrix_1.to_csv(os.path.join(args.save_dir, "cell_line_tissue_pca_distance_1.csv"), index=True)
    pca_distance_matrix_2 = compute_pca_distance(distance_matrix_2, cell_lines_list, tissue_cell_line_dict, args.save_dir)
    pca_distance_matrix_2.to_csv(os.path.join(args.save_dir, "cell_line_tissue_pca_distance_2.csv"), index=True)
    # save the heatmap of the newt distance matrix with altair, the matrix is shape of num_cell_lines x num_tissues
    heatmap_data = pd.DataFrame({
        "Tissue": pca_distance_matrix_1.columns.repeat(pca_distance_matrix_1.shape[0]),
        "Cell_Line_Tissue": np.tile(pca_distance_matrix_1.index, pca_distance_matrix_1.shape[1]),
        "PCA_Distance": pca_distance_matrix_1.values.flatten()
    })

    heatmap = alt.Chart(heatmap_data).mark_rect().encode(
        x=alt.X('Tissue:O', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Cell_Line_Tissue:O', title="Cell Line Tissue of Origin"),
        color='PCA_Distance:Q'
    )
    heatmap.interactive().save(os.path.join(args.save_dir, "pca_cell_line_tissue_distance_heatmap_1.html"))

    heatmap_data = pd.DataFrame({
        "Tissue": pca_distance_matrix_2.columns.repeat(pca_distance_matrix_2.shape[0]),
        "Cell_Line_Tissue": np.tile(pca_distance_matrix_2.index, pca_distance_matrix_2.shape[1]),
        "PCA_Distance": pca_distance_matrix_2.values.flatten()
    })
    heatmap = alt.Chart(heatmap_data).mark_rect().encode(
        x=alt.X('Tissue:O', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Cell_Line_Tissue:O', title="Cell Line Tissue of Origin"),
        color='PCA_Distance:Q'
    )
    heatmap.interactive().save(os.path.join(args.save_dir, "pca_cell_line_tissue_distance_heatmap_2.html"))

    # permutation test
    D1 = pca_distance_matrix_1
    D2 = pca_distance_matrix_2
    p_dict, obs_dict, perm_stats_dict = perm_test_centroid_pair(D1, D2, distance_matrix_1_sub, distance_matrix_2_sub,
                                                                tissue_cell_line_dict, n_perm=5000, two_sided=False, seed=42)
    # Save p-values
    p_values_df = pd.DataFrame.from_dict(p_dict, orient='index', columns=['p_value'])
    p_values_df.to_csv(os.path.join(args.save_dir, "pca_distance_permutation_test_p_values_1_less_than_2.csv"))

    print("PCA distance matrix computed and saved.")

    # Create boxplot
    box_plot(pca_distance_matrix_1, pca_distance_matrix_2, args.save_dir)

if __name__ == "__main__":
    main()