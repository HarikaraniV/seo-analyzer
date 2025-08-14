# image_search.py
import os
import io
import requests
from typing import List
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# ---------- CLIP (image → labels) ----------
_CLIP_MODEL = None
_CLIP_PROCESSOR = None

DEFAULT_LABELS = [
    "engagement ring", "ring", "diamond ring", "gold ring", "necklace", "bracelet",
    "wrist watch", "handbag", "sneakers", "laptop", "smartphone", "headphones",
    "camera", "gaming console", "dress", "jeans", "t-shirt", "sunglasses", "perfume"
]

def _get_clip():
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
        _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _CLIP_MODEL, _CLIP_PROCESSOR

def extract_keywords_from_image(image_file, top_k: int = 3, candidate_labels: List[str] = None) -> List[str]:
    # image_file: streamlit UploadedFile or file-like
    try:
        image = Image.open(image_file).convert("RGB")
    except Exception:
        if hasattr(image_file, "read"):
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            raise

    labels = candidate_labels or DEFAULT_LABELS
    model, processor = _get_clip()

    prompts = [f"a photo of a {lbl}" for lbl in labels]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy().flatten()

    ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
    return [lbl for lbl, _ in ranked[:max(1, top_k)]]

# ---------- SerpApi (labels → URLs) ----------
def search_google_serpapi(query: str, count: int = 10) -> List[str]:
    """
    Returns a list of URLs from SerpApi (Google Search).
    Set SERPAPI_KEY in environment.
    """
    api_key = "5026745ef8aaf752f9804820f325cb058a0927d8890f6d18e7141a02d5c459f5"
    if not api_key:
        print("⚠️ SERPAPI_KEY not set.")
        return []

    try:
        resp = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": query, "num": count, "api_key": api_key},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        urls = []
        for item in data.get("organic_results", []):
            link = item.get("link")
            if link and link.startswith("http"):
                urls.append(link)
        # Deduplicate, keep order
        seen = set()
        uniq = []
        for u in urls:
            if u not in seen:
                uniq.append(u)
                seen.add(u)
        return uniq
    except Exception as e:
        print(f"Error fetching from SerpApi: {e}")
        return []
