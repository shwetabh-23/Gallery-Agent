import pandas as pd
import os

def find_embedding_path(base_folder, subfolder, filename):
    filename = filename[:-4]
    path = os.path.join(base_folder, subfolder, f"{filename}.pt")
    return path if os.path.exists(path) else 'NOT_FOUND'

def process_embeddings(csv_path, embeddings_folder, output_csv_path):
    # Load the original CSV
    df = pd.read_csv(csv_path)

    # Ensure no leading/trailing whitespace in titles
    df['title'] = df['title'].astype(str).str.strip()

    # Prepare new columns for embedding paths
    people_paths = []
    ocr_paths = []
    metadata_paths = []
    caption_paths = []
    for title in df['title']:
        people_paths.append(find_embedding_path(embeddings_folder, 'people', title))
        ocr_paths.append(find_embedding_path(embeddings_folder, 'ocr', title))
        metadata_paths.append(find_embedding_path(embeddings_folder, 'metadata', title))
        caption_paths.append(find_embedding_path(embeddings_folder, 'caption', title))

    # Add new columns to the dataframe
    df['people_embedding'] = people_paths
    df['ocr_embedding'] = ocr_paths
    df['metadata_embedding'] = metadata_paths
    df['caption_embedding'] = caption_paths

    # Save the updated dataframe
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to: {output_csv_path}")

if __name__ == '__main__' : 
    # Example usage:
    process_embeddings("data.csv", "embeddings_split", "data_with_embeddings.csv")
