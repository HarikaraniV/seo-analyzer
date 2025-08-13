import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import logging

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger()

# ------------------------
# Keyword extraction (unchanged)
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
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    filtered = [w for w in words if w not in stopwords]
    common = Counter(filtered).most_common(max_keywords)
    return [word for word, _ in common]

# ------------------------
# TF-IDF Cosine Similarity for keyword relevance
# ------------------------
def check_keyword_relevance_tfidf(page_text, keyword):
    documents = [page_text, keyword.lower()]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return sim_score

# ------------------------
# Meta extraction with relevance check
# ------------------------
def extract_meta_details(url, keyword_input=None):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string if soup.title else ""
        meta_desc = ""
        for meta in soup.find_all("meta"):
            if meta.get("name", "").lower() == "description":
                meta_desc = meta.get("content", "")
                break

        images = soup.find_all("img")
        alt_tag_percent = (len([img for img in images if img.get("alt")]) / len(images) * 100) if images else 0

        page_text = soup.get_text(separator=" ").lower()
        word_count = len(re.findall(r'\b\w+\b', page_text))
        keywords = extract_keywords(page_text, max_keywords=10)

        # Calculate keyword density
        keyword_density = 0
        if keyword_input:
            keyword_occurrences = len(re.findall(rf"\b{re.escape(keyword_input.lower())}\b", page_text))
            keyword_density = (keyword_occurrences / word_count * 100) if word_count > 0 else 0
        else:
            if keywords:
                top_kw = keywords[0]
                keyword_occurrences = len(re.findall(rf"\b{top_kw}\b", page_text))
                keyword_density = (keyword_occurrences / word_count * 100) if word_count > 0 else 0

        # Check keyword relevance using TF-IDF cosine similarity
        keyword_relevant = False
        if keyword_input:
            keyword_lower = keyword_input.lower()
            if keyword_lower not in page_text:
                keyword_relevant = False
            else:
                sim_score = check_keyword_relevance_tfidf(page_text, keyword_input)
                logger.debug(f"Cosine similarity for '{keyword_input}' = {sim_score}")


                threshold = 0.001
                keyword_relevant = sim_score > threshold
            keyword_relevant = bool(keyword_relevant)
        else:
            keyword_relevant = False

        return {
            "Word_Count": word_count,
            "Keyword_Density": round(keyword_density, 2),
            "Keyword_Relevant": keyword_relevant,
            "Meta_Title_Length": len(title),
            "Meta_Desc_Length": len(meta_desc),
            "Alt_Tag_Percent": round(alt_tag_percent, 2),
        }, keywords

    except Exception as e:
        print(f"❌ Error extracting metadata: {e}")
        return None, []

# ------------------------
# Bing News Trends (unchanged)
# ------------------------
def get_global_trending_keywords():
    trending_keywords = []
    try:
        rss_url = "https://www.bing.com/news/search?q=top+trending&format=rss"
        resp = requests.get(rss_url)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "xml")
            titles = soup.find_all("title")
            all_text = " ".join(t.get_text() for t in titles)
            trending_keywords = extract_keywords(all_text, max_keywords=10)
    except Exception as e:
        print(f"⚠️ Failed to fetch Bing trends: {e}")
    return trending_keywords

# ------------------------
# Compare user site keywords with global trends (unchanged)
# ------------------------
def compare_with_my_keywords(my_keywords, global_trends):
    return list(set(my_keywords) & set(global_trends))

# ------------------------
# PageSpeed Insights (unchanged)
# ------------------------
def get_pagespeed_score(url):
    API_KEY = "AIzaSyBykafQYZR8ReIY9LXFYAWGiYVGSt9RJrU"
    if not API_KEY:
        return None
    try:
        api_url = f"https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url={url}&key={API_KEY}"
        response = requests.get(api_url)
        data = response.json()
        score = data['lighthouseResult']['categories']['performance']['score'] * 100
        return round(score, 2)
    except Exception as e:
        print(f"⚠️ PageSpeed error: {e}")
        return None

# ------------------------
# Analyze full website (unchanged)
# ------------------------
def analyze_website(url, keyword_input=None):
    meta_data, keywords = extract_meta_details(url, keyword_input=keyword_input)
    global_trends = get_global_trending_keywords()
    overlap = compare_with_my_keywords(keywords, global_trends)
    return meta_data, keywords, global_trends, overlap

# ------------------------
# Predict SEO score (unchanged)
# ------------------------
def predict_seo_score(model, meta_data):
    # list only the features your model was trained on:
    feature_keys = ['Word_Count', 'Keyword_Density', 'Meta_Title_Length', 'Meta_Desc_Length', 'Alt_Tag_Percent']
    filtered_data = {k: meta_data[k] for k in feature_keys}
    df = pd.DataFrame([filtered_data])
    return model.predict(df)[0]


def load_model():
    model_path = os.path.join("model", "seo_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
