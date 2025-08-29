import os
import numpy as np
import pandas as pd
from sentence_embedding import load_embedding_model, get_embeddings
from faiss_utils import search_faiss
from collections import defaultdict

def retrieve_images_from_query(
    query,
    embedding_types=["people_embedding", "ocr_embedding", "caption_embedding", "metadata_embedding"],
    top_k_each=150,
    final_top_k=10,
    base_path="faiss_indices",
    csv_path="data_with_embeddings.csv"
):
    # Step 1: Load sentence embedding model and encode query
    model, tokenizer = load_embedding_model()
    query_embedding = get_embeddings([query], model, tokenizer).numpy().astype("float32")

    # Step 2: Load CSV for metadata lookup
    df = pd.read_csv(csv_path, dtype={'title': str})
    row_map = {row['title']: row for _, row in df.iterrows()}

    # Step 3: Search all FAISS indices and accumulate similarity scores
    score_dict = defaultdict(lambda: {"score": 0.0, "sources": set(), "meta": None})

    for emb_type in embedding_types:
        index_path = os.path.join(base_path, f"{emb_type}.index")
        id_map_path = os.path.join(base_path, f"{emb_type}_faiss_id_map.csv")

        node_ids, distances = search_faiss(query_embedding, n=top_k_each, index_path=index_path, id_map_path=id_map_path)

        for node_id, dist in zip(node_ids, distances):
            if pd.isna(node_id) or node_id not in row_map:
                continue

            similarity = 1 / (1 + dist)  # Convert L2 distance to similarity

            if score_dict[node_id]["meta"] is None:
                row = row_map[node_id]
                score_dict[node_id]["meta"] = {
                    "title": row["title"],
                    "image_path": row["image_path"],
                    "caption": row.get("caption_text", ""),
                    "ocr": row.get("ocr_text", ""),
                    "people": row.get("people", "")
                }

            score_dict[node_id]["score"] += similarity
            score_dict[node_id]["sources"].add(emb_type)

    # Step 4: Sort results by total score
    sorted_results = sorted(score_dict.items(), key=lambda x: x[1]["score"], reverse=True)[:final_top_k]

    # Step 5: Format final output
    output = []
    for node_id, data in sorted_results:
        meta = data["meta"]
        output.append({
            "title": meta["title"],
            "image_path": meta["image_path"],
            "similarity_score": round(data["score"], 4),
            "caption": meta["caption"],
            "ocr": meta["ocr"],
            "people": meta["people"],
            "sources": list(data["sources"])
        })

    return output

if __name__ == '__main__':
    query = "Show me a picture where Yashwant and Divyansh are sitting together and playing cards."
    results = retrieve_images_from_query(query)

    print("\n🔍 Final Retrieved Images (Combined Top 10):\n" + "-"*50)
    for item in results:
        print(f"📌 {item['title']} | Score: {item['similarity_score']:.4f} | Sources: {', '.join(item['sources'])}")
        print(f"People: {item['people']}")
        print(f"OCR: {item['ocr']}")
        print(f"Caption: {item['caption']}")
        print(f"Image Path: {item['image_path']}")
        print("-" * 50)