import streamlit as st
from seo_utils import analyze_website, load_model, predict_seo_score, get_pagespeed_score

# Load model
try:
    model = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.set_page_config(page_title="SEO Analyzer", layout="centered")
st.title("🔍 SEO Analyzer Tool")

# Inputs
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
