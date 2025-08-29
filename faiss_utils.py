import os
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
import torch

def build_faiss_for_type(df, embedding_col, id_col='title', output_dir="faiss_indices"):
    os.makedirs(output_dir, exist_ok=True)

    embeddings = []
    id_to_faiss = {}
    dim = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"🔍 Processing {embedding_col}"):
        emb_path = row.get(embedding_col)
        if pd.isna(emb_path) or not os.path.exists(emb_path):
            continue

        emb = torch.load(emb_path)
        if emb.ndim == 2:
            emb = emb[0]  # sentence-transformer style (1, dim)
        if dim is None:
            dim = emb.shape[0]
        embeddings.append(emb.cpu().numpy().astype("float32"))

        id_to_faiss[len(embeddings) - 1] = row[id_col]

    if not embeddings:
        print(f"❌ No embeddings found for {embedding_col}")
        return

    # Build FAISS index
    embeddings_np = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)

    # Save index
    index_path = os.path.join(output_dir, f"{embedding_col}.index")
    faiss.write_index(index, index_path)

    # Save mapping
    map_path = os.path.join(output_dir, f"{embedding_col}_faiss_id_map.csv")
    map_df = pd.DataFrame({
        "faiss_id": list(id_to_faiss.keys()),
        "image_id": list(id_to_faiss.values())
    })
    map_df.to_csv(map_path, index=False)

    print(f"✅ FAISS index built for '{embedding_col}' with {len(embeddings)} embeddings.")
    print(f"📦 Saved to {index_path}")
    print(f"🗺️ Mapping saved to {map_path}")

def build_all_faiss_indexes(csv_path="data_with_embeddings.csv"):
    df = pd.read_csv(csv_path)
    
    embedding_types = {
        "people_embedding": "people",
        "ocr_embedding": "ocr",
        "caption_embedding": "caption",
        "metadata_embedding": "metadata"
    }

    for col_name, label in embedding_types.items():
        print(f"\n⚙️ Building FAISS for: {label}")
        build_faiss_for_type(df, embedding_col=col_name, id_col="title", output_dir="faiss_indices")

def search_faiss(query_embedding, n=5, index_path="/home/sdh_innovation_poc/innovation_chatbot_with_graph/innovation_graph_2/faiss_index.index", id_map_path="/home/sdh_innovation_poc/innovation_chatbot_with_graph/innovation_graph_2/faiss_id_map.csv"):

    index = faiss.read_index(index_path)
    query_embedding = np.array(query_embedding).reshape(1, -1).astype("float32")

    distances, indices = index.search(query_embedding, n)
    faiss_ids = indices[0]
    distances = distances[0]

    # Load the ID map and convert to dictionary for fast lookup
    id_map = pd.read_csv(id_map_path)
    faiss_to_image = dict(zip(id_map['faiss_id'], id_map['image_id']))

    # Lookup node_ids in the same order as faiss_ids
    image_ids = [faiss_to_image.get(fid, None) for fid in faiss_ids]

    return image_ids, distances

if __name__ == "__main__":
    build_all_faiss_indexes()
