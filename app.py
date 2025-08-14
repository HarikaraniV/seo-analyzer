import os
import streamlit as st
from seo_utils import analyze_website, load_model, predict_seo_score, get_pagespeed_score
from image_search import extract_keywords_from_image, search_bing

st.set_page_config(page_title="SEO Analyzer & Image-Based Product Search", layout="wide")
st.title("SEO Analyzer & Image-Based Product Search")

# ========================
# Section 1: Image → Product URLs
# ========================
st.header("🖼️ Search Product via Image")

uploaded_file = st.file_uploader("Upload an image of the product", type=["png", "jpg", "jpeg"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Analyzing image to detect product keywords..."):
        detected_labels = extract_keywords_from_image(uploaded_file, top_k=3)
    if detected_labels:
        st.success(f"Detected labels: {', '.join(detected_labels)}")

        # Try to search web for each label until we get some results
        all_urls = []
        for label in detected_labels:
            urls = search_bing(label, count=10)
            if urls:
                st.subheader(f"🔗 Websites selling: {label}")
                for u in urls:
                    st.write(u)
                all_urls.extend(urls)
        if not all_urls:
            st.warning("No relevant websites found (or search API key missing).")
    else:
        st.warning("Could not infer a product label from the image.")

st.markdown("---")

# ========================
# Section 2: Website SEO Analyzer
# ========================
# Inputs

try:
    model = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
    
st.title("🔍 SEO Analyzer Tool")
url = st.text_input("Enter website URL (include http:// or https://):")
keyword_input = st.text_input("Enter a keyword to check its density (optional):")

if url:
    with st.spinner("Analyzing website..."):
        meta_data, keywords, global_trends, overlap = analyze_website(url, keyword_input)
        pagespeed_score = get_pagespeed_score(url)

    if meta_data:
        st.subheader("📊 Meta Information")
        st.json(meta_data)

        st.subheader("🔑 Extracted Keywords")
        st.write(", ".join(keywords) if keywords else "No keywords found.")

        st.subheader("🌍 Global Trending Keywords")
        st.write(", ".join(global_trends) if global_trends else "No trending keywords found from Bing News.")

        st.subheader("🔍 Overlapping Keywords with Global Trends")
        st.write(", ".join(overlap) if overlap else "No overlap with global trends.")

        st.subheader("⚡ PageSpeed Insights Score")
        if pagespeed_score is not None:
            st.metric("Performance Score", f"{pagespeed_score}/100")
        else:
            st.warning("Unable to fetch PageSpeed score. Add API key in environment.")

        if keyword_input:
            st.subheader(f"🔑 Relevance for '{keyword_input}'")
            relevance_text = "✅ Keyword is relevant to this page." if meta_data.get("Keyword_Relevant") else "❌ Keyword is NOT relevant or rarely found."
            st.write(relevance_text)


        st.subheader("📈 Predicted SEO Score")
        try:
            seo_score = predict_seo_score(model, meta_data)
            st.success(f"{round(seo_score, 2)} / 100")
        except Exception as e:
            st.error(f"Error predicting SEO score: {e}")
    else:
        st.error("Failed to extract metadata from the provided URL.")
st.markdown("---")
st.caption("Tip: set environment variables RAPIDAPI_KEY, PAGESPEED_API_KEY, and SIMILARWEB_API_KEY to enable extra data.")
