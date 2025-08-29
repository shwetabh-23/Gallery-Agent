import pandas as pd
import os
import torch
from florence import load_florence2_model, generate_detailed_caption
from sentence_embedding import get_embeddings, load_embedding_model
import re
import unicodedata
from tqdm import tqdm

def build_metadata_text(row):
    return "\n".join([
        f"Year: {row['year']}",
        f"Taken Time: {row['taken_time']}"
    ])

def save_embedding(text_list, model, tokenizer, save_path):
    if not text_list or not isinstance(text_list[0], str) or not text_list[0].strip():
        return None  # Skip empty or invalid text
    embedding = get_embeddings(text_list, model, tokenizer)
    torch.save(embedding, save_path)
    return save_path

def clean_text(text: str) -> str:
    """
    Cleans a given text string by:
    - Normalizing unicode characters
    - Removing excessive whitespace
    - Removing non-printable characters
    - Replacing unwanted special characters
    """
    if not isinstance(text, str):
        return ""

    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Replace weird unicode spaces and invisible characters
    text = re.sub(r'[\u200b-\u200f\u202f\u2060\uFEFF]', ' ', text)

    # Remove non-ASCII, keep basic punctuations and accents
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Collapse multiple spaces or newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text
def process_csv_for_embeddings(csv_path, output_dir="embeddings_split", output_csv="final_data_split.csv"):
    # Setup folders
    os.makedirs(output_dir, exist_ok=True)
    people_dir = os.path.join(output_dir, "people")
    ocr_dir = os.path.join(output_dir, "ocr")
    caption_dir = os.path.join(output_dir, "caption")
    metadata_dir = os.path.join(output_dir, "metadata")
    for folder in [people_dir, ocr_dir, caption_dir, metadata_dir]:
        os.makedirs(folder, exist_ok=True)

    # Load models
    florence_model, florence_proc, florence_device, florence_dtype = load_florence2_model()
    embedding_model, embedding_tokenizer = load_embedding_model()

    df = pd.read_csv(csv_path)
    output_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Processing rows"):

        image_path = row['image_path']
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue
        if image_path.endswith('.mp4') or image_path.endswith('.3gp') or image_path.endswith('.m4v') : 
            continue

        title = row['title']
        filename = title[:-4]  # remove .jpg or .png

        if os.path.exists(os.path.join(people_dir, f"{filename}.pt")) or os.path.exists(os.path.join(ocr_dir, f"{filename}.pt")) or os.path.exists(os.path.join(caption_dir, f"{filename}.pt")) or os.path.exists(os.path.join(metadata_dir, f"{filename}.pt")) : 
            continue
        # Generate OCR & Caption
        ocr = (generate_detailed_caption(image_path, "<OCR>", florence_model, florence_proc, florence_device, florence_dtype))
        caption = generate_detailed_caption(image_path, "<DETAILED_CAPTION>", florence_model, florence_proc, florence_device, florence_dtype)
        breakpoint()
        # Text segments
        people_text = clean_text(row['people'])
        ocr_text = clean_text(ocr['<OCR>'].strip())
        caption_text = clean_text(caption['<DETAILED_CAPTION>'].strip())
        metadata_text = clean_text(build_metadata_text(row))
        # Embedding generation  
        people_path = save_embedding([people_text], embedding_model, embedding_tokenizer, os.path.join(people_dir, f"{filename}.pt"))
        ocr_path = save_embedding([ocr_text], embedding_model, embedding_tokenizer, os.path.join(ocr_dir, f"{filename}.pt"))
        caption_path = save_embedding([caption_text], embedding_model, embedding_tokenizer, os.path.join(caption_dir, f"{filename}.pt"))
        metadata_path = save_embedding([metadata_text], embedding_model, embedding_tokenizer, os.path.join(metadata_dir, f"{filename}.pt"))

        # Prepare output row
        output_rows.append({
            "title": title,
            "image_path": image_path,
            "year": row['year'],
            "time_taken": row['taken_time'],
            "people": people_text,
            "ocr_text": ocr_text,
            "caption_text": caption_text,
            "people_embedding": people_path,
            "ocr_embedding": ocr_path,
            "caption_embedding": caption_path,
            "metadata_embedding": metadata_path
        })

    # Write to CSV
    final_df = pd.DataFrame(output_rows)
    final_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✅ Saved embeddings to '{output_dir}', and final CSV to '{output_csv}'")


if __name__ == '__main__':
    process_csv_for_embeddings("data.csv")
