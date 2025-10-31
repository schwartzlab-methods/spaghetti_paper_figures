'''
Correlate the PCA components with the labels
'''
import numpy as np
from sklearn.decomposition import PCA
import argparse

def correlate_pca_with_labels(pca_components, labels):
    """
    Correlate PCA components with labels.
    
    Parameters:
    pca_components (np.ndarray): PCA components.
    labels (np.ndarray): Labels to correlate with.
    
    Returns:
    np.ndarray: Correlation coefficients.
    """
    # Ensure the inputs are numpy arrays
    pca_components = np.array(pca_components)
    labels = np.array(labels)
    
    # Check if the number of samples match
    if pca_components.shape[0] != labels.shape[0]:
        raise ValueError("Number of samples in PCA components and labels must match.")
    
    # Calculate correlation coefficients
    corr = np.corrcoef(pca_components[:, 0], labels)[0, 1]
    print(f"Correlation coefficients: {corr}")

def main():
    parser = argparse.ArgumentParser(description="Correlate PCA components with labels.")
    parser.add_argument("--numpy_file", type=str, required=True, help="Path to the numpy file containing raw features.")
    parser.add_argument("--labels", type=str, required=True, help="Path to the numpy file containing labels.")
    args = parser.parse_args()
    # Load the PCA components and labels
    features = np.load(args.numpy_file)
    # do pca
    pca = PCA(n_components=50)
    pca_components = pca.fit_transform(features)
    var_ex = pca.explained_variance_ratio_[0]
    print(f"Variance explained by PC1 components: {var_ex}")
    labels = np.load(args.labels)

    # Correlate PCA components with labels
    correlate_pca_with_labels(pca_components, labels)

if __name__ == "__main__":
    main()
    


