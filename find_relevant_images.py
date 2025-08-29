import torch
import pandas as pd
import os
from collections import Counter
from sentence_embedding import get_embeddings, load_embedding_model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_all_embeddings(csv_path):
    df = pd.read_csv(csv_path)
    rows, meta_embs, people_embs, ocr_embs, caption_embs = [], [], [], [], []

    for _, row in df.iterrows():
        try:
            # Skip rows with any NaN embedding paths
            if any(pd.isna(row[col]) for col in ['metadata_embedding', 'people_embedding', 'ocr_embedding', 'caption_embedding']):
                continue

            meta = torch.load(row['metadata_embedding'])
            people = torch.load(row['people_embedding'])
            ocr = torch.load(row['ocr_embedding'])
            caption = torch.load(row['caption_embedding'])

            rows.append(row)
            meta_embs.append(meta)
            people_embs.append(people)
            ocr_embs.append(ocr)
            caption_embs.append(caption)

        except Exception as e:
            print(f"⚠️ Error loading embeddings for {row.get('title', 'unknown')}: {e}")
            continue

    df_valid = pd.DataFrame(rows)
    return df_valid, torch.stack(meta_embs), torch.stack(people_embs), torch.stack(ocr_embs), torch.stack(caption_embs)


def get_top_k_indices(query_emb, all_embs, k):
    sims = cosine_similarity(query_emb, all_embs)[0]
    indices = sims.argsort()[-k:][::-1]
    return indices, sims[indices]

def find_similar_images_sets(query_dict, csv_path="final_data_split.csv", top_k=5):
    """
    query_dict should be:
    {
        'metadata': "text string here" or None,
        'people': "text string here" or None,
        'ocr': "text string here" or None,
        'caption': "text string here" or None
    }
    """
    embedding_model, tokenizer = load_embedding_model()
    df = pd.read_csv(csv_path)

    results = {}

    for field in ['metadata', 'people', 'ocr', 'caption']:
        query_text = query_dict.get(field, None)
        col_name = f"{field}_embedding"

        if not query_text or col_name not in df.columns or query_text == '':
            continue  # Skip empty fields or missing columns

        # Filter rows with non-null embeddings for this field
        filtered_df = df[df[col_name].notna()].copy()
        embeddings = []
        valid_indices = []

        for idx, row in filtered_df.iterrows():
            try:
                emb = torch.load(row[col_name])
                embeddings.append(emb)
                valid_indices.append(idx)
            except Exception as e:
                print(f"⚠️ Failed to load {field} embedding for '{row.get('title', 'unknown')}': {e}")

        if not embeddings:
            print(f"⚠️ No valid embeddings found for field: {field}")
            continue

        # Compute similarity
        query_embedding = get_embeddings([query_text], embedding_model, tokenizer).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, torch.stack(embeddings))[0]

        top_indices = similarities.argsort()[-top_k:][::-1]
        top_rows = filtered_df.iloc[top_indices].copy()
        top_rows['similarity'] = similarities[top_indices]

        results[field] = top_rows[['title', 'image_path', 'similarity', 'caption_text', 'ocr_text', 'people']]

    return results

if __name__ == "__main__":
    query = {
        'metadata': "",
        'people': "",
        'ocr': None,  # OCR left empty, won't be searched
        'caption': "a black and white image of three people."
    }

    result_sets = find_similar_images_sets(query, top_k=5)

    for field, results in result_sets.items():
        print(f"\n🔍 Top results for '{field.upper()}':\n")
        for idx, row in results.iterrows():
            print(f"{row['title']} ({row['similarity']:.3f})")
            print(f"Path: {row['image_path']}")
            print(f"Caption: {row['caption_text']}")
            print(f"OCR: {row['ocr_text']}")
            print(f"People: {row['people']}")
            print("-" * 40)
