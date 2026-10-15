'''
Train a tissue type classifier using H&E images and appy that to PCM images
The classifier is a simple random forest model trained on image features extracted from a pretrained foundational model.
'''

import os
import argparse
import numpy as np
import altair as alt
## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

def main():
    parser = argparse.ArgumentParser(description="Train a tissue type classifier")
    parser.add_argument("--feature", type=str, nargs="+", required=True, help="Path to image features and label npz file")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory for saving the result")
    parser.add_argument("--name", type=str, default="tissue_type_classifier", help="Name for the classifier")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    # random seed
    np.random.seed(42)

    features_L = []
    labels_L = []
    for feature_path in args.feature:
        features = np.load(feature_path)["features"]
        labels = np.load(feature_path)["labels"]
        features_L.append(features)
        labels_L.append(labels)
    features = np.concatenate(features_L, axis=0)
    labels = np.concatenate(labels_L, axis=0)

    all_cell_lines = [
        "BT474", "MCF7", "SkBr3",  # Breast
        "Huh7",                    # Liver
        "SKOV3",                    # Ovarian
    ]

    tissue_cell_line_dict = {
        "Breast": ["BT474", "MCF7", "SkBr3"],
        "Liver": ["Huh7"],
        "Ovarian": ["SKOV3"],
    }

    ## select only the tissues and cell lines in the dictionary
    he_selected_idx = [i for i, label in enumerate(labels) if label in tissue_cell_line_dict.keys()]
    he_features = features[he_selected_idx]
    he_labels = labels[he_selected_idx]

    pcm_selected_idx = [i for i, label in enumerate(labels) if label in all_cell_lines]
    pcm_features = features[pcm_selected_idx]
    pcm_labels = labels[pcm_selected_idx]

    cell_line_to_tissue = {
        cell_line: tissue
        for tissue, cell_lines in tissue_cell_line_dict.items()
        for cell_line in cell_lines
    }

    # translate gt pcm labels to tissue labels
    pcm_labels_tissue = np.array([cell_line_to_tissue[label] for label in pcm_labels])

    print("Number breast HE samples:", sum([1 for label in he_labels if label == "Breast"]))
    print("Number liver HE samples:", sum([1 for label in he_labels if label == "Liver"]))
    print("Number ovarian HE samples:", sum([1 for label in he_labels if label == "Ovarian"]))

    print("Number breast PCM samples:", sum([1 for label in pcm_labels_tissue if label == "Breast"]))
    print("Number liver PCM samples:", sum([1 for label in pcm_labels_tissue if label == "Liver"]))
    print("Number ovarian PCM samples:", sum([1 for label in pcm_labels_tissue if label == "Ovarian"]))

    assert len(he_labels) == len(he_features), "Number of samples in H&E features and labels must match"
    assert len(pcm_labels) == len(pcm_features), "Number of samples in PCM features and labels must match"

    # sample min samples from each he label
    min_samples_per_class = min([sum([1 for label in he_labels if label == tissue]) for tissue in tissue_cell_line_dict.keys()])
    sampled_he_features = []
    sampled_he_labels = []
    for tissue in tissue_cell_line_dict.keys():
        tissue_idx = np.where(he_labels == tissue)[0]
        if len(tissue_idx) > min_samples_per_class:
            sampled_idx = np.random.choice(tissue_idx, min_samples_per_class, replace=False)
        else:
            sampled_idx = tissue_idx
        sampled_he_features.append(he_features[sampled_idx])
        sampled_he_labels.append(he_labels[sampled_idx])
    he_features = np.concatenate(sampled_he_features, axis=0)
    he_labels = np.concatenate(sampled_he_labels, axis=0)

    print("Final number breast HE samples:", sum([1 for label in he_labels if label == "Breast"]))
    print("Final number liver HE samples:", sum([1 for label in he_labels if label == "Liver"]))
    print("Final number ovarian HE samples:", sum([1 for label in he_labels if label == "Ovarian"]))

    # standardize features
    from sklearn.preprocessing import QuantileTransformer

    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    he_features = scaler.fit_transform(he_features)
    pcm_features = scaler.fit_transform(pcm_features)

    # Train a LDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    clf.fit(he_features, he_labels)

    # get the label mapping from classifier
    label_mapping = {i: label for i, label in enumerate(clf.classes_)}
    rev_dic = {v: k for k, v in label_mapping.items()}
    print("Label mapping:", label_mapping) 

    # test on PCM features and plot confusion matrix
    pcm_predictions = clf.predict(pcm_features)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    cm = confusion_matrix(pcm_labels_tissue, pcm_predictions, labels=list(tissue_cell_line_dict.keys()), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(tissue_cell_line_dict.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Tissue Type Classifier Confusion Matrix on PCM Images")
    plt.savefig(os.path.join(args.output, f"pcm_class_confusion_matrix_{args.name}.png"))
    plt.close()

    # per-cell line auc curve
    from sklearn.metrics import roc_curve, auc
    import pandas as pd
    pcm_probabilities = clf.predict_proba(pcm_features)
    auc_data = []
    for i, cell_line in enumerate(all_cell_lines):
        tissue = cell_line_to_tissue[cell_line]
        y_true = (pcm_labels_tissue == tissue).astype(int)
        y_scores = pcm_probabilities[:, rev_dic[tissue]]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        auc_data.append({"Cell Line": [cell_line] * len(y_true), "True Label": y_true, "Probability": y_scores})
        print(f"AUC for cell line {cell_line} ({tissue}): {roc_auc:.2f}")
    # save auc data to csv
    auc_df = pd.DataFrame({
        "Cell Line": np.concatenate([d["Cell Line"] for d in auc_data]),
        "True Label": np.concatenate([d["True Label"] for d in auc_data]),
        "Probability": np.concatenate([d["Probability"] for d in auc_data]),
    })
    auc_df.to_csv(os.path.join(args.output, f"pcm_class_auc_data_{args.name}.csv"), index=False)

    # # plot using altair
    import pandas as pd
    cm_df = pd.DataFrame(cm, index=list(tissue_cell_line_dict.keys()), columns=list(tissue_cell_line_dict.keys()))
    cm_df = cm_df.reset_index().melt(id_vars='index')
    cm_df.columns = ['True Label', 'Predicted Label', 'Proportion']
    # plot with x being -45 degree rotated labels
    heatmap = alt.Chart(cm_df).mark_rect().encode(
        x=alt.X('Predicted Label:N', title='Predicted Label', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('True Label:N', title='True Label'),
        color=alt.Color('Proportion:Q', scale=alt.Scale(scheme='blues'), title='Proportion')
    ).interactive()
    heatmap.save(os.path.join(args.output, f"pcm_class_confusion_matrix_{args.name}.html"))
    
    # compute a matrix for the cell line predictions, where x is tissue and y is cell line
    confusion_matrix = {
        (tissue, cell_line): 0
        for tissue in tissue_cell_line_dict.keys()
        for cell_line in all_cell_lines
    }
    for i, cell_line in enumerate(all_cell_lines):
        cell_line_idx = np.where(pcm_labels == cell_line)[0]
        for tissue in tissue_cell_line_dict.keys():
            tissue_idx = np.where(pcm_predictions[cell_line_idx] == tissue)[0]
            proportion = len(tissue_idx) / len(cell_line_idx) if len(cell_line_idx) > 0 else 0
            confusion_matrix[(tissue, cell_line)] = proportion
            print(f"Proportion of {cell_line} predicted as {tissue}: {proportion:.2f}")

    cm_cell_line_df = pd.DataFrame([
        {'Tissue': tissue, 'Cell Line': cell_line, 'Proportion': proportion}
        for (tissue, cell_line), proportion in confusion_matrix.items()
    ])
    # plot
    heatmap = alt.Chart(cm_cell_line_df).mark_rect().encode(
        x=alt.X('Cell Line:N', title='Cell Line', axis=alt.Axis(labelAngle=-45), sort=all_cell_lines),
        y=alt.Y('Tissue:N', title='Tissue'),
        color=alt.Color('Proportion:Q', scale=alt.Scale(scheme='blues'), title='Proportion')
    ).interactive()
    heatmap.save(os.path.join(args.output, f"pcm_class_confusion_matrix_cell_line_{args.name}.html"))

    # plot with matplotlib
    cm_array = np.zeros((len(tissue_cell_line_dict.keys()), len(all_cell_lines)))
    for i, tissue in enumerate(tissue_cell_line_dict.keys()):
        for j, cell_line in enumerate(all_cell_lines):
            cm_array[i, j] = confusion_matrix[(tissue, cell_line)]
    fig, ax = plt.subplots()
    im = ax.imshow(cm_array, cmap=plt.cm.Blues)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(all_cell_lines)), labels=all_cell_lines, rotation=-45)
    ax.set_yticks(np.arange(len(tissue_cell_line_dict.keys())), labels=list(tissue_cell_line_dict.keys()))
    # Loop over data dimensions and create text annotations.
    for i in range(len(tissue_cell_line_dict.keys())):
        for j in range(len(all_cell_lines)):
            text = ax.text(j, i, f"{cm_array[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_title("Tissue Type Classifier Confusion Matrix on PCM Images (by Cell Line)")
    fig.tight_layout()
    plt.savefig(os.path.join(args.output, f"pcm_class_confusion_matrix_cell_line_{args.name}.png"))

    
    print("Tissue type classifier training and evaluation completed.")

if __name__ == "__main__":
    main()
