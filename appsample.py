import streamlit as st
from seo_utils import (
    analyze_website,
    load_model,
    predict_seo_score,
    get_pagespeed_score,
    get_domain_rank,
)

st.set_page_config(page_title="SEO Analyzer (Sample)", layout="centered")
st.title("🔍 SEO Analyzer (Sample)")

url = st.text_input("Enter website URL (include http:// or https://):")
keyword_input = st.text_input("Enter a keyword to check its density (optional):")

if url:
    with st.spinner("Analyzing website..."):
        meta_data, keywords, global_trends, overlap = analyze_website(url, keyword_input)
        pagespeed_score = get_pagespeed_score(url)
        domain_rank = get_domain_rank(url)

    if meta_data:
        st.subheader("📊 Meta Information")
        st.json(meta_data)

        st.subheader("🔑 Extracted Keywords")
        st.write(", ".join(keywords) if keywords else "No keywords found.")

        st.subheader("🌍 Global Trending Keywords")
        st.write(", ".join(global_trends) if global_trends else "No trending keywords found.")

        st.subheader("🔍 Overlapping Keywords with Global Trends")
        st.write(", ".join(overlap) if overlap else "No overlap with global trends.")

        st.subheader("⚡ PageSpeed Insights Score")
        if pagespeed_score is not None:
            st.metric("Performance Score", f"{pagespeed_score}/100")
        else:
            st.info("Unable to fetch PageSpeed score (no API key set).")

        st.subheader("🏆 Domain Rank")
        if domain_rank is not None:
            st.write(domain_rank)
        else:
            st.info("Domain rank unavailable (no API key set).")

        if keyword_input:
            st.subheader(f"📈 Density for '{keyword_input}'")
            st.write(f"{meta_data.get('Keyword_Density', 0)}%")

        st.subheader("📈 Predicted SEO Score")
        try:
            model = load_model()
            seo_score = predict_seo_score(model, meta_data)
            st.success(f"{round(float(seo_score), 2)} / 100")
        except FileNotFoundError:
            st.info("Model file not found at 'model/seo_model.pkl'. Train it or skip this metric.")
        except Exception as e:
            st.error(f"Error predicting SEO score: {e}")
    else:
        st.error("Failed to extract metadata from the provided URL.")
