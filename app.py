# app.py
import os
import streamlit as st
from image_search import extract_keywords_from_image, search_google_serpapi
from seo_utils import analyze_website, load_model, predict_seo_score, get_pagespeed_score
from product_ranker import rank_urls_by_seo

st.set_page_config(page_title="SEO Analyzer + Image Product Finder", layout="wide")
st.title("🛒 Image → Product URLs + 🔍 SEO Ranking")

# Load model once
try:
    model = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# -------- Section 1: Image to URLs ranked by SEO --------
st.header("1) Upload Product Image → Find & Rank Stores by SEO Score")

uploaded = st.file_uploader("Upload a product image (png/jpg)", type=["png", "jpg", "jpeg"])
top_k = st.slider("How many labels to try from image", 1, 5, 3)
max_urls = st.slider("Max URLs to analyze (across all labels)", 5, 30, 12)

if uploaded:
    with st.spinner("Detecting product labels from image..."):
        labels = extract_keywords_from_image(uploaded, top_k=top_k)
    if labels:
        st.success(f"Detected labels: {', '.join(labels)}")

        # Gather URLs from SerpApi for each label
        collected = []
        for lbl in labels:
            urls = search_google_serpapi(lbl, count=max_urls)
            for u in urls:
                if u not in collected:
                    collected.append(u)
            if len(collected) >= max_urls:
                break

        if not collected:
            st.warning("No URLs found from SerpApi (check SERPAPI_KEY).")
        else:
            st.info(f"Analyzing {len(collected)} URLs… this may take a moment.")
            ranked = rank_urls_by_seo(collected[:max_urls], model, keyword_hint=labels[0])

            if not ranked:
                st.warning("No analyzable URLs.")
            else:
                st.subheader("Ranked Results (by SEO Score)")
                for i, r in enumerate(ranked, start=1):
                    with st.expander(f"#{i} — {r['url']}"):
                        st.write(f"**SEO Score (model)**: {round(r['seo_score'], 2) if r['seo_score']==r['seo_score'] else 'N/A'}")
                        st.write(f"**PageSpeed**: {r['pagespeed'] if r['pagespeed'] is not None else 'N/A'}")
                        st.write("**Meta**:", r["meta"])
                        st.write("**Extracted Keywords**:", ", ".join(r["keywords"]) or "—")
                        st.write("**Global Trending Keywords**:", ", ".join(r["global_trends"]) or "—")
                        st.write("**Overlap w/ Trends**:", ", ".join(r["overlap"]) or "—")
    else:
        st.warning("Couldn’t detect labels from the image.")

st.markdown("---")

# -------- Section 2: Manual Website SEO Analyzer --------
st.header("2) Analyze a Single Website")
url = st.text_input("Enter website URL (include http:// or https://):")
keyword_input = st.text_input("Optional: keyword to check density/relevance")

if url:
    with st.spinner("Analyzing website…"):
        meta_data, keywords, global_trends, overlap = analyze_website(url, keyword_input)
        pagespeed_score = get_pagespeed_score(url)

    if meta_data:
        st.subheader("📊 Meta Information")
        st.json(meta_data)

        st.subheader("🔑 Extracted Keywords")
        st.write(", ".join(keywords) if keywords else "No keywords found.")

        st.subheader("🌍 Global Trending Keywords")
        st.write(", ".join(global_trends) if global_trends else "No Bing-trend keywords found.")

        st.subheader("🔍 Overlapping Keywords with Global Trends")
        st.write(", ".join(overlap) if overlap else "No overlap.")

        st.subheader("⚡ PageSpeed Insights Score")
        if pagespeed_score is not None:
            st.metric("Performance Score", f"{pagespeed_score}/100")
        else:
            st.warning("No PageSpeed score (set GOOGLE_PAGESPEED_API_KEY).")

        st.subheader("📈 Predicted SEO Score")
        try:
            seo_score = predict_seo_score(model, meta_data)
            st.success(f"{round(seo_score, 2)} / 100")
        except Exception as e:
            st.error(f"Error predicting SEO score: {e}")
    else:
        st.error("Failed to extract metadata from the provided URL.")

st.markdown("---")
st.caption("Set environment variables: SERPAPI_KEY and GOOGLE_PAGESPEED_API_KEY.")
