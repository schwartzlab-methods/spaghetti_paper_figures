import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import argparse
import os
import altair as alt
## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")


def main(path_1, path_2, path_label, save_dir, extractor_name, exp_name):
    if path_label:
        original = np.load(path_1)
        spaghetti = np.load(path_2)
        cell_type = np.load(path_label)
    else:
        # we are loading a single npz file with both features and labels
        original = np.load(path_1)['features']
        spaghetti = np.load(path_2)['features']
        cell_type = np.load(path_2)['labels']

    print("Numpy files loaded")

    # map cell type names to numbers
    # cell_type_dict = {}
    # for i, cell in enumerate(np.unique(cell_type)):
    #     cell_type_dict[cell] = i
    # cell_type_num = [cell_type_dict[cell] for cell in cell_type]

    # pca
    # pca = PCA(n_components=2)
    pca = PCA(n_components=50)
    embedding_original = pca.fit_transform(original)
    var_ex_original = pca.explained_variance_ratio_
    embedding_spaghetti = pca.fit_transform(spaghetti)
    var_ex_spaghetti = pca.explained_variance_ratio_

    # prep pandas for altair scatter
    df_original = pd.DataFrame({"PC1": embedding_original[:, 0],
                                "PC2": embedding_original[:, 1],
                                "Classes": cell_type})
    df_spaghetti = pd.DataFrame({"PC1": embedding_spaghetti[:, 0],
                                 "PC2": embedding_spaghetti[:, 1],
                                 "Classes": cell_type})
    
    # plot with altair
    scatter1 = alt.Chart(df_original).mark_point().encode(
        x=alt.X("PC1", title=f"PC 1 (Variance Explained: {var_ex_original[0] * 100:.2f})%"),
        y=alt.Y("PC2", title=f"PC 2 (Variance Explained: {var_ex_original[1] * 100:.2f})%"),
        color="Classes:N",
    ).interactive()
    scatter2 = alt.Chart(df_spaghetti).mark_point().encode(
        x=alt.X("PC1", title=f"PC 1 (Variance Explained: {var_ex_spaghetti[0] * 100:.2f})%"),
        y=alt.Y("PC2", title=f"PC 2 (Variance Explained: {var_ex_spaghetti[1] * 100:.2f})%"),
        color="Classes:N",
    ).interactive()
    scatter1.save(os.path.join(save_dir, f"feature_pca_{extractor_name}_{exp_name}_original.html"))
    scatter2.save(os.path.join(save_dir, f"feature_pca_{extractor_name}_{exp_name}_spaghetti.html"))

    # plot with plt
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # scatter1 = ax[0].scatter(embedding_original[:, 0], embedding_original[:, 1], c=cell_type_num, cmap='tab10', s=0.1)
    # ax[0].set_title(extractor_name)
    # scatter2 = ax[1].scatter(embedding_spaghetti[:, 0], embedding_spaghetti[:, 1], c=cell_type_num, cmap='tab10', s=0.1)
    # ax[1].set_title(f"Spaghetti + {extractor_name}")
    # handles, labels = scatter2.legend_elements()
    # ax[1].legend(handles, np.unique(cell_type), title="Class", loc='center left', bbox_to_anchor=(1.04, 0.5))
    # final_save_dir = os.path.join(save_dir, f"feature_pca_{extractor_name}_{exp_name}.png")
    # plt.savefig(final_save_dir, bbox_inches="tight")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Plotting PCA")
    argparser.add_argument("--save_dir", type=str, help="The directory to save the plot")
    argparser.add_argument("--path_1", type=str, help="Path to the original feature extractor features")
    argparser.add_argument("--path_2", type=str, help="Path to SPAGHETTI features")
    argparser.add_argument("--path_label", default=None, type=str, help="Path to the labels for both")
    argparser.add_argument("--extractor", type=str, help="name of the feature extractor")
    argparser.add_argument("--exp_name", type=str, help="name of the experiment")
    args = argparser.parse_args()
    main(args.path_1, args.path_2, args.path_label, args.save_dir, args.extractor, args.exp_name)

