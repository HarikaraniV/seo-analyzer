import os
import re
import tldextract
import requests
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter
import joblib

# ------------------------
# Keyword extraction
# ------------------------
def extract_keywords(text, max_keywords=10):
    stopwords = set([
        'the','and','to','of','a','in','for','is','on','that','with','as','are','was',
        'at','by','an','be','this','it','from','or','which','but','not','have','has',
        'they','their','you','we','can','will','your','all','more','about','up','our',
        'us','may','ago','hours','just','also','said','news','time','day','how','what',
        'when','where','why','who','whom','while','because','been','if','into','over',
        'after','before','during','so','than','too','very','then','there','here','such',
        'only','each','both','any','some','few','many','every','its','like','other',
        'out','off','again','new','old','via','according','report','reports','online','based'
    ])
    words = re.findall(r'\b[a-z]{3,}\b', (text or "").lower())
    filtered = [w for w in words if w not in stopwords]
    common = Counter(filtered).most_common(max_keywords)
    return [word for word, _ in common]

# ------------------------
# Meta extraction
# ------------------------
def extract_meta_details(url, keyword_input=None):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SEO-Analyzer/1.0)"}
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        meta_desc = ""
        for meta in soup.find_all("meta"):
            if meta.get("name", "").lower() == "description":
                meta_desc = (meta.get("content") or "").strip()
                break

        images = soup.find_all("img")
        img_with_alt = [img for img in images if img.get("alt")]
        alt_tag_percent = (len(img_with_alt) / len(images) * 100) if images else 0

        page_text = soup.get_text(separator=" ")
        word_count = len(re.findall(r'\b\w+\b', page_text))
        keywords = extract_keywords(page_text, max_keywords=10)

        keyword_density = 0.0
        if word_count > 0:
            chosen_kw = None
            if keyword_input:
                chosen_kw = keyword_input.strip().lower()
            elif keywords:
                chosen_kw = keywords[0]
            if chosen_kw:
                occurrences = len(re.findall(rf"\b{re.escape(chosen_kw)}\b", page_text.lower()))
                keyword_density = (occurrences / word_count * 100)

        return {
            "Word_Count": int(word_count),
            "Keyword_Density": round(float(keyword_density), 2),
            "Meta_Title_Length": int(len(title or "")),
            "Meta_Desc_Length": int(len(meta_desc or "")),
            "Alt_Tag_Percent": round(float(alt_tag_percent), 2),
        }, keywords

    except Exception as e:
        print(f"❌ Error extracting metadata: {e}")
        return None, []

# ------------------------
# Global trending keywords (simple, keyless via RSS)
# ------------------------
def get_global_trending_keywords():
    trending_keywords = []
    try:
        rss_url = "https://www.bing.com/news/search?q=top+trending&format=rss"
        resp = requests.get(rss_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "xml")
            titles = soup.find_all("title")
            all_text = " ".join(t.get_text() for t in titles)
            trending_keywords = extract_keywords(all_text, max_keywords=10)
    except Exception as e:
        print(f"⚠️ Failed to fetch Bing trends: {e}")
    return trending_keywords

def compare_with_my_keywords(my_keywords, global_trends):
    return list(set(my_keywords or []) & set(global_trends or []))

# ------------------------
# PageSpeed (optional)
# ------------------------
def get_pagespeed_score(url):
    API_KEY = "AIzaSyBykafQYZR8ReIY9LXFYAWGiYVGSt9RJrU"  # Google PageSpeed Insights API key
    if not API_KEY:
        return None
    try:
        api_url = f"https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url={url}&key={API_KEY}"
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        score = data['lighthouseResult']['categories']['performance']['score'] * 100
        return round(float(score), 2)
    except Exception as e:
        print(f"⚠️ PageSpeed error: {e}")
        return None

# ------------------------
# Domain rank (optional)
# Tries Similarweb (official API) if SIMILARWEB_API_KEY is set.
# Returns a small dict or None.
# ------------------------
def get_domain_rank(url):
    key = os.getenv("SIMILARWEB_API_KEY")
    if not key:
        return None
    try:
        # Extract domain
        ext = tldextract.extract(url)
        domain = ".".join(part for part in [ext.domain, ext.suffix] if part)

        # Example endpoint (adjust per your Similarweb subscription)
        # Here we attempt traffic & rank overview
        sw_url = f"https://api.similarweb.com/v1/website/{domain}/global-rank/global-rank"
        resp = requests.get(sw_url, params={"api_key": key}, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            # Data shape depends on plan; attempt a compact read
            # Fallback to raw JSON if structure unknown
            if isinstance(data, dict) and "GlobalRank" in data:
                return {"domain": domain, "global_rank": data.get("GlobalRank")}
            return {"domain": domain, "raw": data}
        else:
            print(f"Similarweb API error {resp.status_code}: {resp.text}")
            return None
    except Exception as e:
        print(f"⚠️ Domain rank error: {e}")
        return None

# ------------------------
# Analyze website (master)
# ------------------------
def analyze_website(url, keyword_input=None):
    meta_data, keywords = extract_meta_details(url, keyword_input=keyword_input)
    global_trends = get_global_trending_keywords()
    overlap = compare_with_my_keywords(keywords, global_trends)
    return meta_data, keywords, global_trends, overlap

# ------------------------
# Model: load & predict
# ------------------------
def load_model():
    model_path = os.path.join("model", "seo_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

def predict_seo_score(model, meta_data):
    if not meta_data:
        raise ValueError("meta_data is None; cannot predict.")
    feature_keys = ['Word_Count', 'Keyword_Density', 'Meta_Title_Length', 'Meta_Desc_Length', 'Alt_Tag_Percent']
    filtered_data = {k: meta_data.get(k, 0) for k in feature_keys}
    df = pd.DataFrame([filtered_data])
    return model.predict(df)[0]
