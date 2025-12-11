import pandas as pd
import os

# --- Configuration ---
IMG_CSV_PATH = 'data/koniq/koniq10k_scores.csv'
IMG_FOLDER_PATH = 'data/koniq/images'  # <--- Verify this matches your unzipped folder name
TXT_JSON_PATH = 'data/text_data/tweets.json'

def check_data():
    print("--- InsightLens Data Check ---\n")

    # 1. Check Image CSV
    if os.path.exists(IMG_CSV_PATH):
        df_img = pd.read_csv(IMG_CSV_PATH)
        print(f"✅ Image CSV Loaded: {len(df_img)} rows")
    else:
        print("❌ Image CSV missing")
        return # Stop if critical file missing

    # 2. Check Actual Image Files
    if os.path.exists(IMG_FOLDER_PATH):
        # List all files in the directory
        files = os.listdir(IMG_FOLDER_PATH)
        # Filter for likely image files (jpg, png) just to be safe
        img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        count = len(img_files)
        print(f"✅ Image Files Found: {count}")
        
        # Comparison logic
        if count == len(df_img):
            print("   🌟 Perfect Match! (CSV rows == Image files)")
        else:
            print(f"   ⚠️ Warning: Mismatch. CSV has {len(df_img)}, Folder has {count}.")
    else:
        print(f"❌ Image Folder not found at: {IMG_FOLDER_PATH}")

    # 3. Check Text Data
    if os.path.exists(TXT_JSON_PATH):
        try:
            df_txt = pd.read_json(TXT_JSON_PATH, lines=True)
        except ValueError:
            df_txt = pd.read_json(TXT_JSON_PATH)
            
        print(f"\n✅ Text Data Loaded: {len(df_txt)} rows")
        print(f"   Columns: {df_txt.columns.tolist()}")
        
        # Quick check for empty tweets
        empty_tweets = df_txt['tweet'].isnull().sum()
        if empty_tweets > 0:
             print(f"   ⚠️ Note: {empty_tweets} rows have empty text.")
    else:
        print("\n❌ Text JSON missing")

if __name__ == "__main__":
    check_data()