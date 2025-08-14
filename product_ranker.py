# product_ranker.py
from typing import List, Dict, Any
from seo_utils import analyze_website, predict_seo_score, get_pagespeed_score
import math

def rank_urls_by_seo(urls: List[str], model, keyword_hint: str = None) -> List[Dict[str, Any]]:
    """
    For each URL:
      - analyze SEO (meta, keywords, overlaps)
      - get PageSpeed
      - predict SEO score (model)
    Returns list sorted by SEO score DESC.
    """
    results = []
    for url in urls:
        try:
            meta_data, keywords, global_trends, overlap = analyze_website(url, keyword_hint)
            if not meta_data:
                continue

            # pagespeed (may be None if no API key)
            ps_score = get_pagespeed_score(url)

            # predict SEO score
            try:
                seo_score = float(predict_seo_score(model, meta_data))
            except Exception:
                seo_score = float("nan")

            results.append({
                "url": url,
                "seo_score": seo_score,
                "pagespeed": ps_score,
                "meta": meta_data,
                "keywords": keywords,
                "global_trends": global_trends,
                "overlap": overlap,
            })
        except Exception as e:
            print(f"Error analyzing {url}: {e}")

    # sort: highest SEO score first; NaNs at bottom
    def sort_key(item):
        val = item.get("seo_score", float("nan"))
        return (-val if isinstance(val, (int, float)) and not math.isnan(val) else float("inf"))

    results.sort(key=sort_key)
    return results
