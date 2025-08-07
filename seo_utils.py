# seo_utils.py
import requests
from bs4 import BeautifulSoup
import re

def fetch_html(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        return response.text if response.status_code == 200 else None
    except:
        return None

def get_word_count(text):
    return len(re.findall(r'\w+', text))

def get_keyword_density(text, keyword):
    words = re.findall(r'\w+', text.lower())
    keyword = keyword.lower()
    return (words.count(keyword) / len(words)) * 100 if words else 0

def extract_meta_tags(soup):
    title = soup.title.string if soup.title else ""
    description = soup.find("meta", attrs={"name": "description"})
    return len(title), len(description["content"]) if description and description.get("content") else 0

def check_alt_tags(soup):
    images = soup.find_all("img")
    if not images: return 100.0
    with_alt = [img for img in images if img.get("alt")]
    return (len(with_alt) / len(images)) * 100

def get_seo_metrics(url, keyword):
    html = fetch_html(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=' ', strip=True)

    return {
        "Word_Count": get_word_count(text),
        "Keyword_Density": round(get_keyword_density(text, keyword), 2),
        "Meta_Title_Length": extract_meta_tags(soup)[0],
        "Meta_Desc_Length": extract_meta_tags(soup)[1],
        "Alt_Tag_Percent": round(check_alt_tags(soup), 2)
    }
