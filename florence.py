import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import os

def load_florence2_model(model_name="microsoft/Florence-2-large"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    return model, processor, device, torch_dtype

def generate_detailed_caption(image_path, task_prompt, model, processor, device, torch_dtype):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ File not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    
    
    inputs = processor(
        text=task_prompt,
        images=image,
        return_tensors="pt"
    ).to(device, torch_dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

if __name__ == "__main__":
    image_path = 'D:\ML-Projects\gallery_agent\data\IMG-20150914-WA0002.jpg'
    task_prompt = "<DETAILED_CAPTION>"
    task_prompt = "<OCR>"
    model, processor, device, torch_dtype = load_florence2_model()
    caption = generate_detailed_caption(image_path, task_prompt, model, processor, device, torch_dtype)

    print("\n📝 Detailed Caption Output:")
    print(caption)
