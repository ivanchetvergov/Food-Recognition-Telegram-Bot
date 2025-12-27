import os
import argparse
import numpy as np
import joblib
from sklearn.decomposition import PCA

def fit_pca(emb_dir, out_path, n_components=128):
    emb_path = os.path.join(emb_dir, 'embeddings.npy')
    if not os.path.exists(emb_path):
        print(f"Embeddings not found at {emb_path}")
        return
    
    embeddings = np.load(emb_path)
    print(f"Loaded embeddings of shape {embeddings.shape}")
    
    n_samples = embeddings.shape[0]
    n_components = min(n_components, n_samples)
    
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(pca, out_path)

    np.save(os.path.join(emb_dir, 'all_reduced.npy'), reduced_embeddings)
    
    print(f"PCA fitted and saved to {out_path}. Reduced embeddings saved to {emb_dir}/all_reduced.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    
    fit_pca(args.emb_dir, args.out, args.dim)
