import os
import time
import requests
import traceback
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def simulate_copy_image_address_and_download(photo_url: str, save_path: str):
    """
    Simulates 'copy image address' in a public Google Photos link using Selenium.
    Downloads the high-res image to the given save_path.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=chrome_options)

    try:
        print(f"Opening Google Photos URL: {photo_url}")
        driver.get(photo_url)

        # Wait for images to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "img"))
        )

        # Find all images
        all_imgs = driver.find_elements(By.TAG_NAME, "img")

        # Try to pick the largest image (usually has fife.usercontent in src)
        candidate_imgs = [
            img.get_attribute("src")
            for img in all_imgs
            if img.get_attribute("src") and
               "photos.fife.usercontent.google.com" in img.get_attribute("src")
        ]

        if not candidate_imgs:
            raise Exception("❌ No valid image found — photo may require login.")

        # Simulate 'copy image address' by using the src
        real_img_url = candidate_imgs[0]
        print(f"✅ Image source found: {real_img_url}")

        response = requests.get(real_img_url, timeout=10)
        response.raise_for_status()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(response.content)

        print(f"✅ Image downloaded to {save_path}")

    except Exception:
        print(f"[ERROR] Could not download image:\n{traceback.format_exc()}")
    finally:
        driver.quit()

if __name__ == '__main__' : 
    simulate_copy_image_address_and_download(
    photo_url='https://photos.google.com/photo/AF1QipNIfTK98j2r_reRR5h1U4gREm7VkdemYBBjQDZ-',
    save_path=r'myphoto.jpg')
