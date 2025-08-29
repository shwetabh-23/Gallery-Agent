from collections import defaultdict
import os
import pandas as pd
import numpy as np
from sentence_embedding import load_embedding_model, get_embeddings
from faiss_utils import search_faiss

def retrieve_top_k_per_embedding(
    queries_by_embedding,
    model, tokenizer,
    top_k_each=200,
    base_path="faiss_indices",
    csv_path="data_with_embeddings.csv"
):
    """
    queries_by_embedding: dict like {
        "people_embedding": "query about people",
        "ocr_embedding": "query about OCR",
        ...
    }

    Returns:
        result_dict: {
            "people_embedding": [ {title, similarity, image_path, ...}, ... ],
            ...
        }
    """
    
    df = pd.read_csv(csv_path, dtype={'title': str})
    row_map = {row['title']: row for _, row in df.iterrows()}

    result_dict = {}

    for emb_type, query in queries_by_embedding.items():
        if not query.strip():
            continue

        query_embedding = get_embeddings([query], model, tokenizer).numpy().astype("float32")
        index_path = os.path.join(base_path, f"{emb_type}.index")
        id_map_path = os.path.join(base_path, f"{emb_type}_faiss_id_map.csv")

        node_ids, distances = search_faiss(query_embedding, n=top_k_each, index_path=index_path, id_map_path=id_map_path)

        results = []
        for node_id, dist in zip(node_ids, distances):
            if pd.isna(node_id) or node_id not in row_map:
                continue

            row = row_map[node_id]
            similarity = 1 / (1 + dist)

            results.append({
                "title": row["title"],
                "image_path": row["image_path"],
                "caption": row.get("caption_text", ""),
                "ocr": row.get("ocr_text", ""),
                "people": row.get("people", ""),
                "similarity": round(similarity, 4),
                "source": emb_type
            })

        result_dict[emb_type] = results

    return result_dict

def fuse_embedding_results(results_dict, final_top_k=10, multi_source_boost=0.3):
    """
    Combines per-embedding results into a final top-k list by boosting scores 
    of images that appear in multiple embedding types.

    Args:
        results_dict: output from retrieve_top_k_per_embedding
        final_top_k: number of top final results to return
        multi_source_boost: score boost factor for multi-source appearances

    Returns:
        fused_results: list of dicts containing combined results
    """
    fused = defaultdict(lambda: {"score": 0.0, "sources": set(), "meta": None})

    # Accumulate scores and track sources
    for emb_type, results in results_dict.items():
        for row in results:
            node_id = row["title"]
            if fused[node_id]["meta"] is None:
                fused[node_id]["meta"] = row

            fused[node_id]["score"] += row["similarity"]
            fused[node_id]["sources"].add(emb_type)

    # Apply boosting for multi-source appearance
    for node_id, info in fused.items():
        source_count = len(info["sources"])
        boost = 1 + multi_source_boost * (source_count - 1)
        info["score"] *= boost

    # Sort and extract top-k
    sorted_results = sorted(fused.items(), key=lambda x: x[1]["score"], reverse=True)[:final_top_k]

    final_output = []
    for node_id, data in sorted_results:
        meta = data["meta"]
        final_output.append({
            "title": meta["title"],
            "image_path": meta["image_path"],
            "similarity_score": round(data["score"], 4),
            "caption": meta["caption"],
            "ocr": meta["ocr"],
            "people": meta["people"],
            "sources": list(data["sources"])
        })

    return final_output

if __name__ == '__main__' : 
    queries = {
        "people_embedding": "Divyansh and Yashwant",
        "ocr_embedding": "",
        "caption_embedding": "playing cards together",
        "metadata_embedding": "sep 2021"
    }
    model, tokenizer = load_embedding_model()
    per_type_results = retrieve_top_k_per_embedding(queries, model, tokenizer)
    final_results = fuse_embedding_results(per_type_results)
    breakpoint()
    for item in final_results:
        print(item["title"], item["similarity_score"], item["sources"])
