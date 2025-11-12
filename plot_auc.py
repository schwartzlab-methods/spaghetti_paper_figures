'''
Plot AUC curve from a list of CSV with columns "Cell Line", "True Label", and "Probability"
combine multiple CSV files, colour each line by "Cell Line", and style the lines by the source CSV file.
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse

import altair as alt
## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

def plot_auc(csv_files, labels, save_path):
    plt.figure(figsize=(8, 6))
    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)
        for cell_line in df['Cell Line'].unique():
            cell_df = df[df['Cell Line'] == cell_line]
            y_true = cell_df['True Label']
            y_scores = cell_df['Probability']
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label=f'{label} - {cell_line} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, 'auc_plot.png'), dpi=300)
    plt.close()

def plot_auc_altair(csv_files, labels, save_path):
    all_data = []
    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)
        df['Source'] = label
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)

    def compute_roc_auc(data):
        fpr, tpr, _ = roc_curve(data['True Label'], data['Probability'])
        roc_auc = auc(fpr, tpr)
        return pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'AUC': roc_auc})

    roc_data = combined_df.groupby(['Source', 'Cell Line']).apply(compute_roc_auc).reset_index()

    chart = alt.Chart(roc_data).mark_line().encode(
        x='FPR',
        y='TPR',
        color='Cell Line:N',
        strokeDash='Source:N',
        tooltip=['Source', 'Cell Line', 'AUC']
    ).interactive()

    chart.save(os.path.join(save_path, 'auc_plot.html'))

def main():
    parser = argparse.ArgumentParser(description='Plot AUC curves from CSV files.')
    parser.add_argument('--csv_files', nargs='+', required=True, help='List of CSV files containing true labels and probabilities.')
    parser.add_argument('--labels', nargs='+', required=True, help='Labels for each CSV file.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the AUC plot.')
    args = parser.parse_args()

    plot_auc(args.csv_files, args.labels, args.save_path)
    plot_auc_altair(args.csv_files, args.labels, args.save_path)

if __name__ == '__main__':
    main()

