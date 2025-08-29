import os
import pandas as pd
import requests
from urllib.parse import urlparse

# Path to your CSV file
csv_path = r'D:\ML-Projects\gallery_agent\all_photos_metadata.csv'

# Path to the parent folder that contains year-based subfolders
parent_folder = r'C:\ML Projects\photos\photos'

# Read CSV
df = pd.read_csv(csv_path)

# Loop through rows where image_path is 'NOT_FOUND'
for idx, row in df.iterrows():
    if row['image_path'] == 'NOT_FOUND':
        year = str(row['year'])
        url = row['url']

        # Extract image filename from URL
        image_filename = row['title']
        breakpoint()
        # Construct target directory and file path
        target_dir = os.path.join(parent_folder, year)
        os.makedirs(target_dir, exist_ok=True)
        save_path = os.path.join(target_dir, image_filename)

        # Download if file doesn't already exist
        if not os.path.exists(save_path):
            try:
                print(f"Downloading {image_filename} to {save_path}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                with open(save_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
