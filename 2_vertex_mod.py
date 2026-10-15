!pip install google-cloud-aiplatform

from google.colab import auth
auth.authenticate_user()

import vertexai
from vertexai.generative_models import GenerativeModel, Image, Part
import json
import requests
import pandas as pd

project_id = '...'
vertexai.init(project=project_id, location='us-central1')

# Initialize model
model = GenerativeModel('gemini-2.5-flash')

from google.colab import drive
drive.mount('/content/drive')

# Verify which model my project is allowed to talk to
test_ids = ["gemini-2.5-flash", "gemini-3.1-flash-lite", "gemini-1.5-flash"]

for tid in test_ids:
    try:
        m = GenerativeModel(tid)
        res = m.generate_content("test")
        print(f"✅ SUCCESS: {tid}")
        break
    except Exception as e:
        print(f"❌ FAILED: {tid} | Error: {str(e)[:50]}...")

API_KEY = 'api_key'
BASE_URL = "https://civitai.com/api/v1/images"
headers = {
        'Authorization': f'Bearer {API_KEY}'
    }

# Define a function for Gemini to analyze and return the result of an image
def analyze_image(image_id):
    query_params = {
        'imageId': image_id
    }

    # Reset img_url to ensure AI fetch the correct image
    img_url = None

    # 1. Get image bytes for Gemini to analyze
    response = requests.get(BASE_URL, headers=headers, params=query_params)
    response.raise_for_status()

    items = response.json().get('items', [])

    if not items:
        raise Exception(f"No image found for ID {image_id}")

    img_url = items[0]['url']
    print(f"DEBUG: AI is actually downloading: {img_url}")
    img_data = requests.get(img_url).content
    img = Part.from_data(data=img_data, mime_type='image/png')

    # 2. Define the policy prompt
    prompt = """
    Act as a Trust & Safety specialist. Analyze this image.
    Return a JSON object with these exact keys:
    {
      "is_flagged": boolean,
      "primary_category": "none/sexual/violence/suggestive/dangerous acts/other",
      "confidence_score": float (0-1),
      "reasoning": "one sentence explanation"
    }
    """

    # 3. Call the model
    model_result = model.generate_content([img, prompt])
    #print(model_result)

    # 4. Parse results
    raw_result = model_result.text.strip().replace('```json', '').replace('```', '')
    result = json.loads(raw_result)
    #print(result)
    return result

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Civitai Image Safety Analysis/civitai-image-url.csv')
df.tail()

print(f"Shape of df: {df.shape}")

ghost_list = []

for idx in df['image_id']:
  headers = {"Authorization": f"Bearer {API_KEY}"}
  query_params = {
      "imageId": idx
  }
  response = requests.get(BASE_URL, headers=headers, params=query_params)

  data = response.json()
  items = data.get('items', [])

  if not items:
      ghost_list.append(idx)
      print(f"No image found with ID: {idx}")
  else:
      print(f"Image found with ID: {idx}")
      continue

print("  Finished scanning missing items.")

print(ghost_list)
print(f"\nThere are {len(ghost_list)} ghosted images out of {len(df)}. Ghost rate is {len(ghost_list)/len(df)*100:.2f}%.")

# Purge all broken links or deleted assets.

df = df[~df['image_id'].isin(ghost_list)]
df.shape

results = []

for index, row in df.iterrows():
  img_id = row['image_id']
  print(f"Analyzing row {index} with image_id {img_id}")

  try:
      verdict = analyze_image(img_id)
      verdict['image_id'] = str(img_id) # Use str() for integer ID to avoid JSON error
      results.append(verdict)
      #print(f"✅ Saved: {verdict}")
  except Exception as e:
      print(f"❌ FAILED: {img_id} | Error: {e}...")

print(results)

df_result = pd.DataFrame(results)
print(df_result.head())

print(f"\nShape of df_result: {df_result.shape}")

df_result.to_json('/content/drive/MyDrive/Colab Notebooks/Civitai Image Safety Analysis/vertex-results.json', orient='records', indent=4)
df_result.to_csv('/content/drive/MyDrive/Colab Notebooks/Civitai Image Safety Analysis/vertex-results.csv', index=False)
print("Success! File saved to Drive.")



