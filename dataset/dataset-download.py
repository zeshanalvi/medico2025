
print("Dataset Loading...")
from datasets import load_dataset
#ds = load_dataset("SimulaMet-HOST/Kvasir-VQA")
ds_train = load_dataset("SimulaMet/Kvasir-VQA-x1", split="train")
ds_test = load_dataset("SimulaMet/Kvasir-VQA-x1", split="test")
print("Dataset Loaded")

print(ds_train)
print(ds_test)

from PIL import Image
import requests
import os

def preprocess_example(example):
    # Set the directory where images will be saved
    download_dir = "D:\\datasets\\medico_test\\"

    # Create the directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Get the image URL and determine the local path
    image_url = example["image"]
    image_name = image_url.split("/")[-1]
    image_path = os.path.join(download_dir, image_name)

    # Check if the image has already been downloaded
    if not os.path.exists(image_path):
        # Download the image
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the image to the specified directory
        with open(image_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

    # Pack features
    return image_path

print("Loading Images...")
image_cache = {}
#train_data = ds_train.map(preprocess_example)
val_data = ds_test.map(preprocess_example)
print("Images Loaded")