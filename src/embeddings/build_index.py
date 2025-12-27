import os
import argparse
import numpy as np
import faiss

def build_index(emb_npy, out_path):
    if not os.path.exists(emb_npy):
        print(f"Reduced embeddings not found at {emb_npy}")
        return
    
    embeddings = np.load(emb_npy).astype('float32')
    print(f"Loaded reduced embeddings of shape {embeddings.shape}")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    faiss.write_index(index, out_path)
    
    print(f"FAISS index built and saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-npy", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    
    build_index(args.emb_npy, args.out)
