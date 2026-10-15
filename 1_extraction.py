import requests
import pandas as pd
import time

# For mounting
from google.colab import drive

# For greyscale
from PIL import Image, ImageFilter
from io import BytesIO
import matplotlib.pyplot as plt

drive.mount('/content/drive')

"""### Query parameter documentation: https://developer.civitai.com/site/reference/images"""

API_KEY = 'api_key'
BASE_URL = "https://civitai.com/api/v1/images"
headers = {"Authorization": f"Bearer {API_KEY}"}
# Define target category and sample size: {browsing level: size}
TARGET_CONFIG = {
    1: 300,
    2: 300,
    4: 300,
    8: 300,
    16: 300
}
LIMIT_PER_PAGE = 200

results_dict = {} # Make it a dictionary rather than df so that it can hold unique image IDs only.
ghost_list = []

for level, size in TARGET_CONFIG.items():
  print(f"Fetching images for browsing level {level}...")
  next_page_url = BASE_URL
  page = 1
  # Track items saved specifically for this current loop's level
  level_count = 0

  # The loop terminates if we hit our size target OR if the platform runs out of pages
  while level_count < size and next_page_url:
      print(f"Fetching images for page {page}...")
      query_params ={
          "limit": LIMIT_PER_PAGE,
          "sort":"Newest",
          #"nsfw": False
          "browsingLevel": level
      }

      try:
        if next_page_url == BASE_URL:
          response = requests.get(next_page_url, headers=headers, params=query_params)
        else:
          response = requests.get(next_page_url, headers=headers)

        data = response.json()
        items = data.get('items', [])

        if not items:
          print(f" No more items returned by API for level {level}. Terminating level early.")
          break

        for img in items:
          result = list(results_dict.values())
          level_count = len([i for i in result if i['browsing_level']==level])

          if level_count >= size:
            break

          if img.get('browsingLevel')!=level:
            continue

          img_id = img['id']

          response2 = requests.get(BASE_URL, headers=headers, params={"imageId": img_id})
          items2 = response2.json().get('items', [])
          # 2nd Check: ghost items
          if not items2:
            print(f"Ghosted item detected: {img_id}. Skipping.")
            ghost_list.append(img_id)
            continue

          # Only add image if it's a new ID
          if img_id not in results_dict:
            results_dict[img_id] = {
                'image_id': img['id'],
                #'url': img['url'],
                'browsing_level': img['browsingLevel'],
                'nsfw': img['nsfw'],
                'created_at': img['createdAt'],
                'stats': img['stats']
            }
            print(f"Saved {img_id} to results_dict.")

        # Re-calculate our current tier's exact count directly from the dictionary
        result = list(results_dict.values())
        level_count = len([i for i in result if i['browsing_level']==level])

        # --- PAGINATION BLOCK ---
        metadata = data.get('metadata', [])
        next_page_url = metadata.get('nextPage')
        page += 1
        time.sleep(0.1) # Protect the code from rate limits

      except Exception as e:
        print(f"Failed to fetch page: {page}: {e}")
        # If a page fails completely, break out to prevent an endless request loop
        break

  print(f"Finished fetching images for browsing level {level}. Total {level_count} items fetched.")

print(results_dict.values())
print(f"\nLength of results_dict: {len(results_dict)}.\n")
print(list(results_dict.values()))
print(f"Length of ghost_list: {len(ghost_list)}")

results = list(results_dict.values())
df = pd.DataFrame(results)

# Verification: Print how many images we got for each level
print(df.head())
print("\nfinal dataset summary:")
print(df[['browsing_level','nsfw']].value_counts(normalize=True))

print(f"Duplicates in image ID: {df.duplicated('image_id').sum()}")

df.to_csv('/content/drive/MyDrive/Colab Notebooks/Civitai Image Safety Analysis/civitai-image-url.csv', index=False)

# Testing space: try fetching a single image
response = requests.get("https://civitai.com/api/v1/images?limit=1&nsfwLevel=None")
data = response.json()

print(data.keys())
print(data['items'][0])
print(data['items'][0]['id'])

# Create a function to call the image in safe view modes
def safe_view(image_id, mode="blur-gray"):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    query_params = {
        "imageId": image_id
    }

    response = requests.get(BASE_URL, headers=headers, params=query_params)

    if response.status_code != 200:
        print(f"Failed to fetch image with ID: {image_id}")
        return

    data = response.json() # Get image metadata
    items = data.get('items', [])

    if not items:
        print(f"No image found with ID: {image_id}")
        return

    img_url = items[0]['url'] # Get url from image metadata

    img_response = requests.get(img_url) # Get the actual image bytes from that URL

    img = Image.open(BytesIO(img_response.content))

    if mode == "grayscale":
        img = img.convert("L") # Converts to Black & White
        plt.imshow(img, cmap='gray')
    elif mode == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=20)) # Heavy blur
        plt.imshow(img)
    elif mode == "blur-gray":
        img = img.convert("L") # Converts to Black & White
        img = img.filter(ImageFilter.GaussianBlur(radius=20)) # Heavy blur
        plt.imshow(img, cmap='gray')

    plt.axis('off')
    plt.show()

# View a flagged image safely
safe_view(23526611, mode="blur-gray")







