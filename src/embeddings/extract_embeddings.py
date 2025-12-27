import os
import argparse
import torch
import pandas as pd
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def extract_embeddings(images_dir, out_dir, model_name):
    os.makedirs(out_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    embeddings = []
    metadata = []
    
    for img_file in tqdm(image_files):
        img_path = os.path.join(images_dir, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            embeddings.append(image_features.cpu().numpy().flatten())
            metadata.append({'image_id': img_file, 'image_path': img_path})
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            
    embeddings = np.array(embeddings)
    np.save(os.path.join(out_dir, 'embeddings.npy'), embeddings)
    
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(os.path.join(out_dir, 'metadata.csv'), index=False)
    
    print(f"Extracted {len(embeddings)} embeddings and saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    args = parser.parse_args()
    
    extract_embeddings(args.images_dir, args.out_dir, args.model)
