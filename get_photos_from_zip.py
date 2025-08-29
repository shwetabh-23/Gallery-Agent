import os
import zipfile
import json
import csv
import requests
from tqdm import tqdm
from datetime import datetime
import shutil

# === CONFIGURATION ===
ZIP_FOLDER = "D:\ML-Projects\gallery_agent\data\zips"                 # folder containing 33 zip files
WORK_DIR = "C:\ML Projects\photos\extracted_zips"             # where zips will be extracted
OUTPUT_DIR = "C:\ML Projects\photos\photos"      # where images will be stored
CSV_PATH = "all_photos_metadata.csv"

# === UTILITY FUNCTIONS ===

def extract_all_zips(zip_folder, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    for zip_file in tqdm(os.listdir(zip_folder), desc="🔓 Extracting ZIPs"):
        if zip_file.endswith('.zip'):
            zip_path = os.path.join(zip_folder, zip_file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

def find_json_files(root_dir):
    json_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.json') and not file.lower().endswith('.mp4.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def extract_metadata(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    title = data.get("title", "")
    views = data.get("imageViews", "")
    taken = data.get("photoTakenTime", {}).get("formatted", "")
    timestamp = data.get("photoTakenTime", {}).get("timestamp", "")
    year = "unknown"

    if timestamp:
        year = datetime.utcfromtimestamp(int(timestamp)).year

    # Prefer geoDataExif if present
    geo = data.get("geoDataExif", data.get("geoData", {}))
    latitude = geo.get("latitude", "")
    longitude = geo.get("longitude", "")

    device = data.get("googlePhotosOrigin", {}).get("mobileUpload", {}).get("deviceType", "")
    local_folder = data.get("googlePhotosOrigin", {}).get("mobileUpload", {}).get("deviceFolder", {}).get("localFolderName", "")

    # ✅ Extract people
    people_list = data.get("people", [])
    people = ", ".join([p.get("name", "") for p in people_list if "name" in p]) if people_list else ""

    url = data.get("url", "")

    return {
        "title": title,
        "year": str(year),
        "taken_time": taken,
        "views": views,
        "latitude": latitude,
        "longitude": longitude,
        "device": device,
        "local_folder": local_folder,
        "people": people,
        "url": url
    }

def create_year_folder(base_dir, year):
    path = os.path.join(base_dir, str(year))
    os.makedirs(path, exist_ok=True)
    return path

def download_image(url, save_path):
    try:
        if os.path.exists(save_path):
            return True  # already downloaded
        response = requests.get(url + "=d")  # =d for direct download
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False

def move_image_from_json(json_path, title, year_folder):
    """
    Copy the image file next to the JSON into the year_folder.
    Returns the new image path or status string.
    """
    image_path = os.path.join(os.path.dirname(json_path), title)
    dest_path = os.path.join(year_folder, title)
    os.makedirs(year_folder, exist_ok=True)

    if os.path.exists(image_path):
        try:
            if not os.path.exists(dest_path):
                shutil.copy2(image_path, dest_path)
            return dest_path
        except Exception as e:
            print(f"⚠️ Error copying {image_path} to {dest_path}: {e}")
            return "COPY_FAILED"
    else:
        return "NOT_FOUND"


def process_all_jsons(json_files, output_image_dir, csv_output_path):
    rows = []
    for json_file in tqdm(json_files, desc="📦 Processing JSONs"):
        try:
            metadata = extract_metadata(json_file)
            year = metadata['year']
            title = metadata['title']

            year_folder = create_year_folder(output_image_dir, year)
            image_path = move_image_from_json(json_file, title, year_folder)
            metadata['image_path'] = image_path

            rows.append(metadata)
        except Exception as e:
            print(f"❌ Error processing {json_file}: {e}")

    # Define all fields (no json_path now)
    fieldnames = [
        "title", "year", "taken_time", "views",
        "latitude", "longitude", "device", "local_folder",
        "people", "url", "image_path"
    ]

    with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ CSV written to {csv_output_path}")

# === MAIN WORKFLOW ===

def main():
    print("🚀 Starting Takeout Processing")
    # extract_all_zips(ZIP_FOLDER, WORK_DIR)

    # This assumes structure: extracted/Takeout/Google Photos/*
    google_photos_dir = os.path.join(WORK_DIR, "Takeout", "Google Photos")
    breakpoint()
    if not os.path.exists(google_photos_dir):
        # In some zips, Google Photos might be inside individual folders
        google_photos_dir = WORK_DIR

    json_files = find_json_files(google_photos_dir)
    print(f"🔍 Found {len(json_files)} JSON metadata files")

    process_all_jsons(json_files, OUTPUT_DIR, CSV_PATH)
    print("✅ All Done!")

if __name__ == '__main__':
    main()
