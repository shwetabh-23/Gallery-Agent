import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

def load_phi3() : 
    # Use 4-bit quantization config (can also try 8-bit if 4-bit doesn't work well on CPU)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,  # Use float32 for CPU
        bnb_4bit_quant_type="nf4"
    )

    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="auto",  # Let HF decide device (CPU fallback)
        quantization_config=bnb_config,
        trust_remote_code=False
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    return model, tokenizer

def get_phi3_inference(messages, model, tokenizer) : 
    # Inference pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Generation arguments
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    # Generate and print
    output = pipe(messages, **generation_args)
    response = (output[0]['generated_text'])
    return response

if __name__ == '__main__' : 

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
        {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    ]
    model, tokenizer = load_phi3()
    response = get_phi3_inference(messages, model, tokenizer)
    print(response)
    breakpoint()