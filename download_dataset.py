import os
from simple_image_download import simple_image_download

# Folder where images will be saved
save_dir = os.path.join(os.getcwd(), "dataset_raw")
os.makedirs(save_dir, exist_ok=True)

# Create downloader instance (lowercase!)
downloader = simple_image_download.simple_image_download()

# Search queries
queries = [
    "moss",
    "healthy grass lawn",
    "brown grass spots",
    "water puddles lawn",
    "dry soil lawn"
]

# Download images
for query in queries:
    print(f"Downloading: {query}")
    downloader.download(query, limit=150)

print("✅ Download complete! Images saved in dataset_raw folder.")
