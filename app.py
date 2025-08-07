import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import joblib
import lightgbm as lgb
from urllib.parse import urlparse

# 🔐 Your Google PageSpeed API Key (replace with actual key)
API_KEY = "AIzaSyBykafQYZR8ReIY9LXFYAWGiYVGSt9RJrU"

# 📦 Load pre-trained LightGBM model
model = joblib.load("model/seo_model.pkl")

# 🔍 Function to fetch meta details using BeautifulSoup
def extract_meta_details(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string if soup.title else ""
        meta_desc = ""
        for meta in soup.find_all("meta"):
            if "name" in meta.attrs and meta.attrs["name"].lower() == "description":
                meta_desc = meta.attrs.get("content", "")
                break

        images = soup.find_all("img")
        img_with_alt = [img for img in images if img.get("alt")]
        alt_tag_percent = (len(img_with_alt) / len(images)) * 100 if images else 0

        word_count = len(soup.get_text().split())
        keyword_density = 1.5  # Placeholder or derived from analysis

        return {
            "Word_Count": word_count,
            "Keyword_Density": keyword_density,
            "Meta_Title_Length": len(title),
            "Meta_Desc_Length": len(meta_desc),
            "Alt_Tag_Percent": round(alt_tag_percent, 2),
        }
    except Exception as e:
        st.error(f"❌ Failed to extract meta details: {e}")
        return None

# 📊 Function to fetch PageSpeed score
def fetch_pagespeed_score(url):
    api_url = f"https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url={url}&key={API_KEY}"
    try:
        response = requests.get(api_url)
        data = response.json()
        score = data['lighthouseResult']['categories']['performance']['score'] * 100
        return round(score, 2)
    except Exception as e:
        st.warning(f"⚠️ Could not fetch PageSpeed score: {e}")
        return None

# 🌐 Streamlit Web App
st.title("🔍 SEO Website Analyzer")
url_input = st.text_input("Enter Website URL (with https://)", placeholder="https://example.com")

if st.button("Analyze"):
    if url_input:
        st.info("⏳ Fetching SEO data...")
        meta_data = extract_meta_details(url_input)
        if meta_data:
            st.success("✅ Meta details fetched successfully!")
            st.write(meta_data)

            score = fetch_pagespeed_score(url_input)
            if score is not None:
                st.write(f"🚀 Google PageSpeed Score: **{score}/100**")

            # Predict SEO score
            df = pd.DataFrame([meta_data])
            predicted_score = model.predict(df)[0]
            st.subheader(f"🔢 Predicted SEO Score: {round(predicted_score, 2)} / 100")

    else:
        st.warning("Please enter a valid URL.")
