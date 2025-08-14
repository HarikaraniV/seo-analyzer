import os
import io
import requests
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ------------------------
# Global CLIP variables
# ------------------------
_clip_model = None
_clip_processor = None

# ------------------------
# Product label list
# ------------------------
DEFAULT_LABELS = [
    "engagement ring", "necklace", "wrist watch", "handbag", "sneakers", "laptop",
    "smartphone", "headphones", "camera", "gaming console", "dress", "jeans",
    "t-shirt", "sunglasses", "perfume", "coffee maker", "microwave", "refrigerator",
    "sofa", "office chair", "bicycle", "helmet", "football", "yoga mat", "drone",
]

# ------------------------
# Lazy CLIP loader
# ------------------------
def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor

# ------------------------
# Image to keywords
# ------------------------
def extract_keywords_from_image(image_file, top_k=3, candidate_labels=None):
    """
    Returns top_k product labels inferred from the image using CLIP zero-shot classification.
    image_file: bytes-like or file-like object from Streamlit uploader.
    """
    try:
        # Try opening directly
        image = Image.open(image_file).convert("RGB")
    except Exception:
        # If stream-like, read bytes first
        if hasattr(image_file, "read"):
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            raise

    labels = candidate_labels or DEFAULT_LABELS
    model, processor = _get_clip()

    # Create text prompts for CLIP
    prompts = [f"a photo of a {lbl}" for lbl in labels]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape [1, num_labels]
        probs = logits_per_image.softmax(dim=1).cpu().numpy().flatten()

    ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
    return [lbl for lbl, _ in ranked[:max(1, top_k)]]

# ------------------------
# Bing RapidAPI Search
# ------------------------
# ------------------------
# Bing RapidAPI Search
# ------------------------
def search_bing(query, count=10):
    """
    Uses SerpApi's Google Search API to get a list of result URLs for the query.
    """
    import requests
    
    api_key = "5026745ef8aaf752f9804820f325cb058a0927d8890f6d18e7141a02d5c459f5"
    params = {
        "engine": "google",    # Search engine type
        "q": query,            # Search query
        "num": count,          # Number of results
        "api_key": api_key     # Your SerpApi key
    }
    
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        response.raise_for_status()
        results = response.json()

        websites = []
        for item in results.get("organic_results", []):
            if "link" in item:
                websites.append(item["link"])
        
        return websites
    except Exception as e:
        print(f"Error fetching search results from SerpApi: {e}")
        return []
