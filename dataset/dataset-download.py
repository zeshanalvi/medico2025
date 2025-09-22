import os
import requests
from PIL import Image
from io import BytesIO
from datasets import load_dataset
import pandas as pd

# 1. Define the local paths and dataset information
dataset_name = "SimulaMet/Kvasir-VQA-x1"
split_name = "train"
output_dir = "D:\\datasets\\me2025\\"
images_dir = os.path.join(output_dir, "images")
csv_file_path = os.path.join(output_dir, "dataset_info.csv")

# Create the output directories if they don't exist
os.makedirs(images_dir, exist_ok=True)

# 2. Load the dataset from Hugging Face
print(f"Loading dataset '{dataset_name}' with split '{split_name}'...")
dataset = load_dataset(dataset_name, split=split_name)

# 3. Create a new list to store the modified data
processed_data = []

# 4. Iterate through the dataset, download images, and process data
print("Processing and downloading images. This may take some time...")
for i, row in enumerate(dataset):
    try:
        # Get the image URL and the desired filename from the dataset row
        image_url = row['image']
        img_id = row['img_id']
        image_path = os.path.join(images_dir, image_url.split("/")[-1])

        # Download the image from the URL
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        # Save the image to the local file system
        with open(image_path, 'wb') as f:
            f.write(response.content)

        # Create a new dictionary for the row data, replacing the URL with the local path
        new_row = {
            'img_id': img_id,
            'image_path': image_path,
            'complexity': row['complexity'],
            'question': row['question'],
            'answer': row['answer'],
            'original': row['original'],
            'question_class': row['question_class'],
            'img_id': row['img_id']
        }
        processed_data.append(new_row)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image {img_id}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for row {i}: {e}")

# 5. Convert the processed data to a pandas DataFrame and save to CSV
if processed_data:
    df = pd.DataFrame(processed_data)
    df.to_csv(csv_file_path, index=False)
    print(f"\nSuccessfully downloaded images and saved dataset info to '{csv_file_path}'")
else:
    print("No data was processed. Check for download errors.")

print("Process completed.")