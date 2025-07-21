import numpy as np

def normalize_embedding(Y):
    """Zero-center and unit-scale the embedding."""
    Y_centered = Y - np.mean(Y, axis=0)
    Y_scaled = Y_centered / np.sqrt(np.mean(np.sum(Y_centered**2, axis=1)))
    return Y_scaled

def compute_edge_weights(Y, edge_index):
    """Compute pairwise Euclidean distances for edges."""
    output =  np.linalg.norm(Y[edge_index[:, 0]] - Y[edge_index[:, 1]], axis=1)
    output = output**2+0.05
    loss = output / (output + 1)
    return loss

def soft_jaccard_similarity(Y1, Y2, edge_index):
    """
    Computes the Soft Jaccard Similarity and Distance between two embeddings.
    
    Parameters:
    - Y1, Y2: (n, d) numpy arrays, two embeddings
    - edge_index: (m, 2) numpy array of fixed edge set E

    Returns:
    - similarity: float
    - distance: float
    """
    # Step 1: Normalize embeddings
    Y1_norm = normalize_embedding(Y1)
    Y2_norm = normalize_embedding(Y2)
    
    # Step 2: Compute edge weights
    W1 = compute_edge_weights(Y1_norm, edge_index)
    W2 = compute_edge_weights(Y2_norm, edge_index)

    # Step 3: Soft Jaccard computation
    denom = W1 + W2
    min_frac = np.minimum(W1, W2) / denom
    max_frac = np.maximum(W1, W2) / denom

    # Handle any zero division cases (when both W1 and W2 are zero)
    mask = denom != 0
    J = np.sum(min_frac[mask]) / np.sum(max_frac[mask])

    # Step 4: Convert to distance
    distance = 1 - J
    return J, distance

def cosine_similarity(Y1,Y2,edge_index):
    """
    Computes the Cosine Similarity and Distance between two embeddings.
    
    Parameters:
    - Y1, Y2: (n, d) numpy arrays, two embeddings
    - edge_index: (m, 2) numpy array of fixed edge set E

    Returns:
    - similarity: float
    - distance: float
    """
    # Step 1: Normalize embeddings
    Y1_norm = normalize_embedding(Y1)
    Y2_norm = normalize_embedding(Y2)
    # Y1_norm = Y1
    # Y2_norm = Y2
    
    # Step 2: Compute edge weights
    W1 = compute_edge_weights(Y1_norm, edge_index)
    W2 = compute_edge_weights(Y2_norm, edge_index)

    # Step 3: Cosine Similarity computation
    dot_product = np.sum(W1 * W2)
    norm_product = np.linalg.norm(W1) * np.linalg.norm(W2)
    
    # Handle any zero division cases (when both W1 and W2 are zero)
    if norm_product == 0:
        similarity = 0
        distance = 1
    else:
        similarity = dot_product / norm_product
        distance = 1 - similarity

    return similarity, distance
