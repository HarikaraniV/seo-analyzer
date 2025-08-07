import pandas as pd
import numpy as np
import os

# Number of rows to generate
NUM_SAMPLES = 10000

# Set random seed for reproducibility
np.random.seed(42)

# Generate random features
word_count = np.random.randint(300, 2501, size=NUM_SAMPLES)
keyword_density = np.round(np.random.uniform(0.5, 3.0, size=NUM_SAMPLES), 2)
meta_title_length = np.random.randint(30, 71, size=NUM_SAMPLES)
meta_desc_length = np.random.randint(70, 181, size=NUM_SAMPLES)
alt_tag_percent = np.random.randint(30, 101, size=NUM_SAMPLES)

# Simulate SEO score (this can be modified later with a trained model)
seo_score = (
    0.2 * (word_count / 2500) * 100 +
    0.25 * (keyword_density / 3.0) * 100 +
    0.15 * (meta_title_length / 70) * 100 +
    0.2 * (meta_desc_length / 180) * 100 +
    0.2 * (alt_tag_percent / 100) * 100 +
    np.random.normal(0, 5, size=NUM_SAMPLES)  # noise
)

seo_score = np.clip(seo_score, 0, 100).astype(int)

# Create DataFrame
df = pd.DataFrame({
    "Word_Count": word_count,
    "Keyword_Density": keyword_density,
    "Meta_Title_Length": meta_title_length,
    "Meta_Desc_Length": meta_desc_length,
    "Alt_Tag_Percent": alt_tag_percent,
    "SEO_Score": seo_score
})

# Save to CSV
os.makedirs("data", exist_ok=True)
df.to_csv("data/seo_dataset.csv", index=False)
print("✅ SEO dataset generated at: data/seo_dataset.csv")
