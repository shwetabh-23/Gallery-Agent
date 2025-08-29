import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image

def load_siglip_model(model_name="google/siglip-base-patch16-512", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor, device

def get_siglip_embeddings(image_path, text, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=image, padding="max_length", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.embeddings shape: [batch_size, embedding_dim]
    return outputs.image_embeds[0].cpu().numpy(), outputs.text_embeds[0].cpu().numpy()

if __name__ == '__main__' : 
    model, processor, device = load_siglip_model() 
    image_path = 'D:\ML-Projects\gallery_agent\data\IMG_20180714_190015.jpg'
    text = 'badminton players'
    image_embeddings, text_embeddings = get_siglip_embeddings(image_path, text, model, processor, device)
    breakpoint()