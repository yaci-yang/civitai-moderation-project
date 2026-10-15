from google.colab import drive
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageFilter

drive.mount('/content/drive')

API_KEY = 'api_key'
BASE_URL = "https://civitai.com/api/v1/images"

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

df1 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Civitai Image Safety Analysis/civitai-image-url.csv')
df1.head()

df2 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Civitai Image Safety Analysis/vertex-results.csv')
df2.head()

df = pd.merge(df1, df2, on='image_id', how='inner')
df.head()

print(df1.shape)
print(df2.shape)
print(df.shape)
print(f"There are {df.duplicated(['image_id']).sum()} duplicated image_id.")

df[['nsfw','is_flagged']].value_counts().sort_index(ascending=[True, True])

df['m_accurate'] = ((df['nsfw'] == True) & (df['is_flagged'] == True)) | ((df['nsfw'] == False) & (df['is_flagged'] == False))
df['m_overkill'] = (df['nsfw'] == False) & (df['is_flagged'] == True)
df['m_leakage'] = (df['nsfw'] == True) & (df['is_flagged'] == False)

tp = (df['nsfw']==True) & (df['is_flagged']==True)
fp = (df['nsfw']==False) & (df['is_flagged']==True)
fn = (df['nsfw']==True) & (df['is_flagged']==False)

print(f"\nAI accuracy rate: {df['m_accurate'].sum() / len(df) * 100: .2f}%")
print(f"AI leakage rate: {df['m_leakage'].sum() / len(df) * 100: .2f}%")
print(f"AI overkill rate: {df['m_overkill'].sum() / len(df) * 100: .2f}%")
print(f"AI precision: {tp.sum()/(tp.sum() + fp.sum()) * 100: .2f}%")
print(f"AI recall: {tp.sum()/(tp.sum() + fn.sum()) * 100: .2f}%")

df.to_csv('/content/drive/MyDrive/Colab Notebooks/Civitai Image Safety Analysis/merged_df.csv', index=False)
print("Successfully saved to drive.")

"""
### In which browsing level is AI most and least aligned with human?
"""
print(df.groupby('browsing_level')[['nsfw','is_flagged']].value_counts())

print(f"{round(df.groupby('browsing_level')['m_accurate'].mean(),2)}")

print(f"{round(df.groupby('browsing_level')['m_overkill'].mean(),2)}")
print(f"{round(df.groupby('browsing_level')['m_leakage'].mean(),2)}")

"""
### In which categories AI moderation have the most discrepancies?
"""

print(df[df['m_accurate']==False]['primary_category'].value_counts(ascending=False))

counts = pd.crosstab(
    df['primary_category'],
    df['m_accurate'],
    margins=True,
    margins_name='Total'
)
#print(counts)

plt.figure(figsize=(5, 2))
sns.heatmap(pd.crosstab(df['primary_category'], df['m_accurate']),
            annot=True, fmt='d', cmap='YlGnBu')
plt.title("Discrepancy Heatmap")
plt.show()

pct = pd.crosstab(
    df['primary_category'],
    df['m_accurate'],
    normalize='index'
) * 100
#print(counts)
#print(pct)

comb = counts.copy()
comb['True %'] = pct[True].map("{:.1f}%".format)
comb['False %'] = pct[False].map("{:.1f}%".format)
comb = comb[[True, 'True %', False, 'False %', 'Total']]
print("AI moderation results for each category")
print(comb)

"""
### How confident is AI in moderating the cases?
"""

df_ai_wrong = df[df['m_accurate']==False]
sns.histplot(data=df_ai_wrong, x='confidence_score', multiple='dodge')
