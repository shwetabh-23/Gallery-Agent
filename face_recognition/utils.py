from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def load_face_detector(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Loads the MTCNN face detector.
    """
    mtcnn = MTCNN(image_size=160, margin=0, device=device, keep_all = True)
    return mtcnn

def load_embedding_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Loads the pre-trained FaceNet (InceptionResnetV1) model.
    """
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return facenet

def detect_faces(image_path, mtcnn):
    """
    Loads image and detects faces using MTCNN.
    Returns list of face tensors. Returns empty list if no face found.
    """
    img = Image.open(image_path).convert('RGB')
    faces = mtcnn(img, return_prob=False)

    if faces is None:
        return []

    return faces  # list of tensors


def get_embeddings(faces, facenet, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Computes embeddings for a list of face tensors using the given facenet model.
    Returns a list of flattened numpy embeddings.
    """
    embeddings = []

    for face in faces:
        face = face.unsqueeze(0).to(device)  # (1, 3, 160, 160)
        with torch.no_grad():
            emb = facenet(face).cpu().numpy().flatten()
        embeddings.append(emb)

    return embeddings

from torchvision.transforms.functional import to_pil_image

def save_tensor_as_image(tensor, save_path, mean=None, std=None):
    """
    Converts a normalized tensor (3, H, W) to a PIL image and saves it.
    """
    assert tensor.shape[0] == 3, "Expected 3 channels (C, H, W)"
    
    if mean is not None and std is not None:
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        tensor = tensor * std + mean  # De-normalize

    tensor = torch.clamp(tensor, 0, 1)  # Ensure values in [0,1]
    img = to_pil_image(tensor)
    img.save(save_path)

if __name__ == '__main__' : 
    mtcnn = load_face_detector()
    facenet = load_embedding_model()
    image_path = r'D:\ML-Projects\gallery_agent\data\IMG_20180714_190015.jpg'
    faces = detect_faces(image_path, mtcnn)

    if len(faces) == 0:
        print("No faces detected.")
    else:
        print(f"{len(faces)} face(s) detected.")
        embeddings = get_embeddings(faces, facenet)
        for i, emb in enumerate(embeddings):
            print(f"Embedding {i+1}: {emb[:5]}... (len: {len(emb)})")  # Preview first 5 values