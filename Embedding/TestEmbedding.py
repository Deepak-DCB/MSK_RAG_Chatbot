import numpy as np
import pandas as pd

# Paths
chunks_path = r"C:\Users\Draco\OneDrive\Documents\MSK_Chat\MSKArticlesINDEX\chunks.parquet"
emb_text_path = r"C:\Users\Draco\OneDrive\Documents\MSK_Chat\Embedding\embeddings.npy"
emb_img_path  = r"C:\Users\Draco\OneDrive\Documents\MSK_Chat\Embedding\embeddingsImages.npy"

# Load parquet and embeddings
df = pd.read_parquet(chunks_path)
E  = np.load(emb_text_path)
EI = np.load(emb_img_path)

# Basic counts
print("📊 Parquet rows:", len(df))
print("📊 embeddings.npy (text):", E.shape)
print("📊 embeddingsImages.npy (text_with_images):", EI.shape)

# Sanity check alignment
if len(df) == E.shape[0] == EI.shape[0]:
    print("✅ Row counts aligned across parquet and both embeddings")
else:
    print("⚠️ Mismatch detected!")

# Find empty rows
empty_idx_text = df[df['text'].str.strip() == ''].index.tolist()
empty_idx_img  = df[df['text_with_images'].str.strip() == ''].index.tolist()

print("Empty text row indices:", empty_idx_text)
print("Empty text_with_images row indices:", empty_idx_img)

# Optional: verify those empty rows are filled with placeholder vectors
if empty_idx_text:
    print("\nVector for first empty text row:", E[empty_idx_text[0]][:10], "...")
if empty_idx_img:
    print("Vector for first empty text_with_images row:", EI[empty_idx_img[0]][:10], "...")
