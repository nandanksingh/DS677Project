from datasets import load_dataset
import os
import shutil
import json

# Load the dataset
ds = load_dataset("nlphuji/flickr30k", split="test")

# Create output dir
os.makedirs("flickr30k_train", exist_ok=True)

# Prepare metadata
metadata = []

for i, row in enumerate(ds):
    image = row["image"]
    caption = row["caption"]
    img_filename = f"{i:05d}.jpg"
    
    # Save image
    image.save(f"flickr30k_train/{img_filename}")
    
    # Add to metadata
    metadata.append({"file_name": img_filename, "text": caption})

# Write metadata.jsonl
with open("flickr30k_train/metadata.jsonl", "w") as f:
    for entry in metadata:
        f.write(json.dumps(entry) + "\n")
