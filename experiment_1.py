import os
import scanpy as sc
import anndata as ad
from pacmap import PaCMAP
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


DEFAULT_DATA_PATH = "/usr/xtmp/UBC2025/data/human_cell_atlas.h5ad"

def load_data(data_path):
    """
    Load the example dataset for experiment 1.
    
    Returns:
        ad.AnnData: The loaded AnnData object containing the dataset.
    """
    try:
        adata = sc.read_h5ad(DATA_PATH)
        print(f"Data loaded successfully from {DATA_PATH}")
        return adata
    except FileNotFoundError:
        print(f"File not found: {DATA_PATH}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None
    
    
def fit_pacmap(matrix, n_components=2, n_neighbors=10):
    """
    Fit a PACMAP model to the given frequency matrix.
    
    Args:
        matrix (np.ndarray): The frequency matrix to fit.
        n_components (int): Number of dimensions for the output embedding.
        n_neighbors (int): Number of neighbors to consider in the PACMAP algorithm.
        
    Returns:
        np.ndarray: The fitted PACMAP embedding.
    """

    pacmap = PaCMAP(n_components=n_components, n_neighbors=n_neighbors)
    embedding = pacmap.fit_transform(matrix)
    return embedding
    

def preprocess_data(adata):
    """
    Preprocess the AnnData object by normalizing and log-transforming the data.
    
    Args:
        adata (ad.AnnData): The AnnData object to preprocess.
        
    Returns:
        ad.AnnData: The preprocessed AnnData object.
    """
    sc.pp.highly_variable_genes(adata, n_top_genes=1000) # reason about this. It does mean adjusted variance of some sort. 
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=30, svd_solver='arpack')
    return adata

def plot_embedding(embedding, labels, title, filename):
        """
        Plot and save the PACMAP embedding.

        Args:
        embedding (np.ndarray): The embedding to plot.
        title (str): Title of the plot.
        filename (str): Filename to save the plot.
        """
        print("This is happening")
        if labels is not None:
            if labels.dtype == 'category':
                labels = labels.cat.codes
            else:
                labels = np.array(labels)
        else:
            labels = np.zeros(embedding.shape[0])
        os.makedirs('embedding', exist_ok=True)
        csv_filename = filename.replace('.png', '.csv')
        np.savetxt(f'embedding/{csv_filename}', embedding, delimiter=',')
        print(f"PACMAP embedding shape: {embedding.shape}")
        plt.figure(figsize=(8, 6))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=5, alpha=0.5)
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.savefig(filename)

def main(data_path, sample_size, preprocess, label_column, use_existing_embeddings=True):
    """
    Main function to execute the experiment.
    """
    np.random.seed(42)  # For reproducibility 
    adata = load_data(data_path)
    if adata is not None:
        # print(f"Dataset contains {adata.n_obs} observations and {adata.n_vars} variables.")
        # print(adata)
        freq_matrix = adata.X
        labels = adata.obs[label_column] if label_column in adata.obs else None
      
        if (preprocess is True):
            print("Preprocessing the dataset...")
            processed_dataset = preprocess_data(adata)
            
            subset_idx = np.random.choice(processed_dataset.n_obs, sample_size, replace=False)
            
            print(f"Selected {len(subset_idx)} random samples from the dataset.")
            X_sample = processed_dataset.X[subset_idx].toarray()
            y_sample = labels[subset_idx] if labels is not None else None
            
            if os.path.exists('/usr/xtmp/UBC2025/embedding/Processed_Human_Cell_Atlas.csv') and use_existing_embeddings:
                print("Processed embedding already exists.")
                embedding_completed = np.loadtxt('/usr/xtmp/UBC2025/embedding/Processed_Human_Cell_Atlas.csv', delimiter=',')
            else:
                print("Fitting PACMAP on the processed dataset...")
                embedding_completed = fit_pacmap(X_sample)
                print("Saving the processed embedding...")
                np.savetxt('/usr/xtmp/UBC2025/embedding/Processed_Human_Cell_Atlas.csv', embedding_completed, delimiter=',')
            plot_embedding(embedding_completed, y_sample, title="PACMAP Embedding of Processed Human Cell Atlas", filename="Processed_Human_Cell_Atlas.png")
        else:
            subset_idx = np.random.choice(adata.n_obs, sample_size, replace=False)
            
            print(f"Selected {len(subset_idx)} random samples from the dataset.")
            
            X_sample = freq_matrix[subset_idx].toarray()
            y_sample = labels[subset_idx] if labels is not None else None
            print(f"Frequency matrix shape: {X_sample.shape}")
            if (os.path.exists('/usr/xtmp/UBC2025/embedding/Human_Cell_Atlas_Sample.csv') and use_existing_embeddings):
                print("Embedding directory already exists.")
                embedding = np.loadtxt('/usr/xtmp/UBC2025/embedding/Human_Cell_Atlas_Sample.csv', delimiter=',')
            else:
                embedding = fit_pacmap(X_sample)
            plot_embedding(embedding, y_sample, title="PACMAP Embedding of Human Cell Atlas Sample", filename="Human_Cell_Atlas_Sample.png")
            
    else:
        print("Failed to load dataset.")
        
if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Run PACMAP on Human Cell Atlas dataset.")
    parse.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='Path to the dataset file.')
    parse.add_argument('--sample_size', type=int, default=100_000, help='Number of samples to randomly select from the dataset.')
    parse.add_argument('--preprocess', type=bool, default=False, help='Whether to preprocess the dataset before fitting PACMAP.')
    parse.add_argument('--label_column', type=str, default='cluster_id', help='Column name for cell type labels in the dataset.')
    parse.add_argument('--use_existing', type=bool, default=True, help='Whether to use existing embeddings if available.')
    args = parse.parse_args()
    
    DATA_PATH = args.data_path
    SAMPLE_SIZE = args.sample_size
    PREPROCESS = args.preprocess
    LABEL_COLUMN = args.label_column
    USE_EXISTING = args.use_existing

    if not os.path.exists(DATA_PATH):
        print(f"Data path {DATA_PATH} does not exist. Please provide a valid path.")
    else:
        main(DATA_PATH, SAMPLE_SIZE, PREPROCESS, LABEL_COLUMN, USE_EXISTING)
