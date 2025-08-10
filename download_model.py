import os
import requests
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

# 🔹 Categories to download (Google Images search keywords)
CATEGORIES = [
    "moss in lawn",
    "healthy grass",
    "brown spots in grass",
    "puddles in yard",
    "dry soil"
]

# 🔹 Where to save images
SAVE_DIR = "dataset_raw"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔹 Number of images per category
NUM_IMAGES = 150

# 🔹 Bing Image Search API (Free Tier)
# Get your API key here: https://www.microsoft.com/en-us/bing/apis/bing-image-search-api
BING_API_KEY = "YOUR_BING_API_KEY"
SEARCH_URL = "https://api.bing.microsoft.com/v7.0/images/search"

def download_image(url, path):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                f.write(r.content)
    except Exception as e:
        print(f"[ERROR] {e}")

def fetch_images_for_category(category):
    print(f"[INFO] Searching for '{category}'...")
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": category, "count": NUM_IMAGES, "safeSearch": "Strict"}

    r = requests.get(SEARCH_URL, headers=headers, params=params)
    r.raise_for_status()
    results = r.json().get("value", [])

    category_dir = os.path.join(SAVE_DIR, category.replace(" ", "_"))
    os.makedirs(category_dir, exist_ok=True)

    urls = [img["contentUrl"] for img in results]
    with ThreadPoolExecutor(max_workers=10) as executor:
        for idx, url in enumerate(urls, start=1):
            path = os.path.join(category_dir, f"{idx}.jpg")
            executor.submit(download_image, url, path)

if __name__ == "__main__":
    for category in CATEGORIES:
        fetch_images_for_category(category)
    print("[DONE] All images downloaded successfully!")

