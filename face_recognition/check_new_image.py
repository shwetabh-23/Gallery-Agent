import numpy as np
from utils import detect_faces, get_embeddings, load_face_detector, load_embedding_model, save_tensor_as_image
import torch
from sklearn.metrics.pairwise import cosine_similarity

def load_person_embedding(name, embeddings_dir="saved_embeddings"):
    """
    Load the stored embedding for a person by name.
    Assumes each person has a saved .npy file with their name.
    """
    try:
        embedding = torch.load(f"{embeddings_dir}/{name}_embedding.pt")
        return embedding
    except FileNotFoundError:
        print(f"No saved embedding found for: {name}")
        return None

def is_person_in_image(image_path, name, mtcnn, facenet, threshold=0.75, embeddings_dir="embeddings"):
    """
    Detects faces in an image and compares their embeddings with the target person.
    Returns True if the person is detected based on similarity threshold.
    """
    person_embedding = load_person_embedding(name, embeddings_dir)
    if person_embedding is None:
        return False

    faces = detect_faces(image_path, mtcnn)
    if len(faces) == 0:
        print("No faces detected.")
        return False

    embeddings = get_embeddings(faces, facenet)
    for emb in embeddings:
        similarity = cosine_similarity([emb], [person_embedding])[0][0]
        print('similarity : ',similarity)
        if similarity >= threshold:
            print(f"Match found with similarity: {similarity:.3f}")
            return True

    print("No matching person found in the image.")
    return False

if __name__ == '__main__' : 
    image_path = r"C:\Users\hahis\Downloads\IMG_20191223_204630.jpg"
    name = "vidit"
    mtcnn = load_face_detector()
    facenet = load_embedding_model()
    found = is_person_in_image(image_path, name, mtcnn, facenet)
    print(f"{name} found in image: {found}")