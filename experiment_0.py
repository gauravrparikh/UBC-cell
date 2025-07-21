import os
import scanpy as sc
import anndata as ad
from pacmap import PaCMAP
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import cne

def load_data(data_path):
    """
    Loads an AnnData object from an .h5ad file.

    Returns:
        ad.AnnData: The loaded AnnData object, or None if an error occurs.
    """
    if os.path.exists('preprocessed_data.h5ad'):
        print("Loading preprocessed data from 'preprocessed_data.h5ad'")
        return sc.read_h5ad('preprocessed_data.h5ad')
    try:
        adata = sc.read_h5ad(data_path)
        print(f"Data loaded successfully from {data_path}")
        return adata
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def fit_pacmap(matrix, output_dir, n_components=2, random_seed=42):
    """
    Fit a PaCMAP model to the given matrix and save pair files.
    
    Args:
        matrix (np.ndarray): The data matrix to fit.
        output_dir (str): Directory to save the generated pair files.
        n_components (int): Number of dimensions for the output embedding.
        random_seed (int): The random seed for reproducibility.
        
    Returns:
        np.ndarray: The fitted PaCMAP embedding.
    """
    print(f"Fitting PaCMAP with random_seed={random_seed}...")
    np.random.seed(random_seed)
    model = PaCMAP(n_components=n_components, random_state=random_seed)
    embedding = model.fit_transform(matrix)
    
    # Save the generated pairs to the specified output directory
    pair_dir = os.path.join(output_dir, 'pacmap_generated_pairs')
    os.makedirs(pair_dir, exist_ok=True)
    np.save(os.path.join(pair_dir, f'Pair_neighbors_seed_{random_seed}.npy'), model.pair_neighbors)
    np.save(os.path.join(pair_dir, f'Pair_MNs_seed_{random_seed}.npy'), model.pair_MN)
    np.save(os.path.join(pair_dir, f'Pair_FPs_seed_{random_seed}.npy'), model.pair_FP)
    
    return embedding
    
def fit_other_embeddings(matrix, embedding_method='infonce', n_components=2):
    """
    Fit an alternative embedding method using the CNE library.
    
    Args:
        matrix (np.ndarray): The data matrix to fit.
        embedding_method (str): The loss mode for the CNE model.
        n_components (int): Number of dimensions for the output embedding.
        
    Returns:
        np.ndarray: The fitted embedding.
    """
    print(f"Fitting CNE with loss_mode='{embedding_method}'...")
    
    model = cne.CNE(optimizer="adam", parametric=True, loss_mode=embedding_method)
    embedding = model.fit_transform(matrix.astype(np.float32))
    return embedding

def preprocess_data(adata):
    """
    Preprocess AnnData: log-transform and find highly variable genes.
    
    Args:
        adata (ad.AnnData): The AnnData object to preprocess.
        
    Returns:
        ad.AnnData: The preprocessed AnnData object.
    """
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    adata = adata[:, adata.var['highly_variable']]
    sc.save(adata, 'preprocessed_data.h5ad')
    return adata

def plot_embedding(embedding, labels, title, output_dir, filename_prefix):
    """
    Plot and save the embedding as both a PNG image and a CSV file.

    Args:
        embedding (np.ndarray): The 2D embedding to plot.
        labels (pd.Series or np.ndarray): Labels for coloring the points.
        title (str): Title of the plot.
        output_dir (str): Directory to save the output files.
        filename_prefix (str): Prefix for the output filenames.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embedding to CSV
    csv_path = os.path.join(output_dir, f'{filename_prefix}.csv')
    np.savetxt(csv_path, embedding, delimiter=',')
    print(f"Saved embedding of shape {embedding.shape} to {csv_path}")

    # Prepare labels for plotting
    if labels is not None:
        if hasattr(labels, 'cat'): # Check if it's a pandas categorical series
            plot_labels = labels.cat.codes
        else:
            plot_labels = np.array(labels)
    else:
        plot_labels = np.zeros(embedding.shape[0])

    # Create and save plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=plot_labels, s=5, alpha=0.6, palette='viridis', legend=None)
    plt.title(title, fontsize=16)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    np.savetxt(os.path.join(output_dir, f'{filename_prefix}_labels.csv'), plot_labels, delimiter=',')
    
    plot_path = os.path.join(output_dir, f'{filename_prefix}.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close() # Close the figure to free memory
    print(f"Saved plot to {plot_path}")

def main(args, run_output_dir, random_seed):
    """
    Main function to execute one run of the experiment.
    """
    np.random.seed(random_seed)
    adata = load_data(args.data_path)
    if adata is None:
        print("Failed to load dataset. Aborting run.")
        return

    print(f"Dataset contains {adata.n_obs} observations and {adata.n_vars} variables.")
    
    # Determine data and labels to use based on preprocessing flag
    if args.preprocess:
        print("Preprocessing the dataset...")
        if 'preprocessed_data.h5ad' in os.listdir('.'):
            print("Preprocessed data already exists. Loading it.")
            adata = sc.read_h5ad('preprocessed_data.h5ad')
        else:
            processed_adata = preprocess_data(adata)
        freq_matrix = processed_adata.X
        labels_to_use = processed_adata.obs[args.label_column] if args.label_column in processed_adata.obs else None
        dataset_name_suffix = "Processed_HCA"
    else:
        print("Skipping preprocessing.")
        freq_matrix = adata.X
        labels_to_use = adata.obs[args.label_column] if args.label_column in adata.obs else None
        dataset_name_suffix = "Raw_HCA"
    
    # Subsample the data for this run
    subset_idx = np.random.choice(freq_matrix.shape[0], args.sample_size, replace=False)
    X_sample = freq_matrix[subset_idx].toarray()
    y_sample = labels_to_use.iloc[subset_idx] if labels_to_use is not None else None
    print(f"Selected {len(subset_idx)} random samples. Data matrix shape: {X_sample.shape}")
    
    # --- PaCMAP Embedding ---
    print("\n--- Processing PaCMAP ---")
    pacmap_filename_prefix = f"PaCMAP_{dataset_name_suffix}"
    pacmap_embedding_path = os.path.join(run_output_dir, f"{pacmap_filename_prefix}.csv")
    if os.path.exists(pacmap_embedding_path) and args.use_existing:
        print(f"Loading existing PaCMAP embedding from {pacmap_embedding_path}")
        pacmap_embedding = np.loadtxt(pacmap_embedding_path, delimiter=',')
    else:
        pacmap_embedding = fit_pacmap(X_sample, output_dir=run_output_dir, random_seed=random_seed)
    
    plot_embedding(
        pacmap_embedding, 
        y_sample, 
        title=f"PaCMAP Embedding ({dataset_name_suffix}, Seed: {random_seed})", 
        output_dir=run_output_dir, 
        filename_prefix=pacmap_filename_prefix
    )

    # --- CNE Embeddings (Multiple Methods) ---
    cne_methods = ['infonce', 'ncvis', 'negtsne', 'umap']
    for method in cne_methods:
        print(f"\n--- Processing CNE with method: {method} ---")
        
        cne_filename_prefix = f"CNE_{method}_{dataset_name_suffix}"
        cne_embedding_path = os.path.join(run_output_dir, f"{cne_filename_prefix}.csv")

        if os.path.exists(cne_embedding_path) and args.use_existing:
            print(f"Loading existing CNE ({method}) embedding from {cne_embedding_path}")
            cne_embedding = np.loadtxt(cne_embedding_path, delimiter=',')
        else:
            cne_embedding = fit_other_embeddings(X_sample, embedding_method=method)

        plot_embedding(
            cne_embedding, 
            y_sample, 
            title=f"CNE ({method}) Embedding ({dataset_name_suffix}, Seed: {random_seed})", 
            output_dir=run_output_dir, 
            filename_prefix=cne_filename_prefix
        )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PaCMAP and multiple CNE methods on a single-cell dataset.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the .h5ad dataset file.')
    parser.add_argument('--output_dir', type=str, default='./results', help='Base directory to save all experiment outputs.')
    parser.add_argument('--sample_size', type=int, default=50_000, help='Number of samples to randomly select for each run.')
    parser.add_argument('--preprocess', action='store_true', help='If set, preprocess the dataset before fitting.')
    parser.add_argument('--label_column', type=str, default='cluster_id', help='Column name for cell type labels in .obs.')
    parser.add_argument('--use_existing', action='store_true', help='If set, use existing embeddings if found, otherwise recompute.')
    parser.add_argument('--n_repeats', type=int, default=1, help='Number of times to repeat the experiment with different random seeds.')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"FATAL: Data path '{args.data_path}' does not exist. Please provide a valid path.")
    else:
        for i in range(args.n_repeats):
            # Each run gets a unique seed and a unique output directory
            random_seed = 42 + i
            run_output_dir = os.path.join(args.output_dir, f"run_{i}_seed_{random_seed}")
            os.makedirs(run_output_dir, exist_ok=True)
            
            print("\n" + "="*60)
            print(f"STARTING RUN {i+1}/{args.n_repeats} (Seed: {random_seed})")
            print(f"Output will be saved to: {run_output_dir}")
            print("="*60)
            
            main(args, run_output_dir, random_seed)

            print("\n" + "="*60)
            print(f"FINISHED RUN {i+1}/{args.n_repeats}")
            print("="*60 + "\n")
