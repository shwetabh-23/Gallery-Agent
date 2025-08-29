import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from utils import detect_faces, get_embeddings, load_face_detector, load_embedding_model

def build_face_embedding_dict(csv_path, mtcnn, facenet, output_dir="embeddings"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = df[df['people'].apply(lambda x: isinstance(x, str) and x.strip() != '')]

    embeddings_dict = {}

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = row['image_path']
        people_str = row['people']

        people = [p.strip() for p in people_str.split(',') if p.strip()]
        if len(people) != 1:
            continue  # Skip if more than 1 person in the people column

        person_name = people[0]

        try:
            faces = detect_faces(image_path, mtcnn)
            if len(faces) == 1:
                embeddings = get_embeddings(faces, facenet)
                embeddings = embeddings[0]
            else : continue
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

        embedding_tensor = torch.tensor(embeddings)

        save_path = os.path.join(output_dir, f"{person_name}_embedding.pt")

        if os.path.exists(save_path):
            existing = torch.load(save_path)
            new_avg = (existing + embedding_tensor) / 2
            torch.save(new_avg, save_path)
        else:
            torch.save(embedding_tensor, save_path)

        embeddings_dict[person_name] = save_path

    return embeddings_dict

if __name__ == '__main__' : 
    csv_path = r'D:\ML-Projects\gallery_agent\data_with_embeddings.csv'
    mtcnn = load_face_detector()
    facenet = load_embedding_model()
    build_face_embedding_dict(csv_path, mtcnn, facenet)