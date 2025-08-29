from prompt import build_structured_query_prompt
from faiss_retrieval_2 import retrieve_top_k_per_embedding, fuse_embedding_results
from phi3 import load_phi3, get_phi3_inference
from sentence_embedding import load_embedding_model

def get_recommendations(query, model, tokenizer) : 
    
    prompt = build_structured_query_prompt(query)
    response = get_phi3_inference(prompt, model, tokenizer)
    return response

import re

def extract_structured_fields(text):
    """
    Extract fields like 'people', 'ocr', 'description', 'metadata' from a structured string.

    Example input:
    'people: Yashwant, Divyansh\nocr: \ndescription: sitting together, playing cards\nmetadata: '

    Returns:
        dict: { 'people': ..., 'ocr': ..., 'description': ..., 'metadata': ... }
    """
    pattern = r'(people|ocr|description|metadata):\s*(.*?)\s*(?=(?:people|ocr|description|metadata):|$)'
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)

    output = {k.lower(): v.strip() for k, v in matches}
    
    # Ensure all keys exist
    for key in ["people", "ocr", "description", "metadata"]:
        output.setdefault(key, "")

    return output

def get_image_recommendations(query, phi3_model, phi3_tokenizer, embedding_model, embedding_tokenizer) : 
    response = get_recommendations(query, phi3_model, phi3_tokenizer)
    response_cleaned = extract_structured_fields(response)
    queries = {
        'people_embedding' : response_cleaned['people'], 
        'ocr_embedding' : response_cleaned['ocr'],
        'caption_embedding' : response_cleaned['description'],
        'metadata_embedding' : response_cleaned['metadata']
    }

    per_type_results = retrieve_top_k_per_embedding(queries, embedding_model, embedding_tokenizer)
    final_results = fuse_embedding_results(per_type_results)

    for item in final_results:
        print(item["title"], item["similarity_score"], item["sources"])

    return [item['image_path'] for item in final_results]

if __name__ == '__main__' : 

    query = "pictures of rio and shwetabh"
    phi3_model, phi3_tokenizer = load_phi3()
    embedding_model, embedding_tokenizer = load_embedding_model()
    images = get_image_recommendations(query, phi3_model, phi3_tokenizer, embedding_model, embedding_tokenizer)
    breakpoint()