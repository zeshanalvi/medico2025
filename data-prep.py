import requests
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
import torch

samples=0


def read_image(img_path):
    response = requests.get(img_path)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    # Step 2: Define transforms (resize, convert to tensor, normalize, etc.)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # adjust size for your model
        transforms.ToTensor(),           # convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])
    # Step 3: Apply transforms
    image = transform(img)
    # Step 4: Add batch dimension and move to device
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #image1 = image.unsqueeze(0).to(device)
    return image


def prepare_dataset():
    return 


#print(image1.shape)  # should be [1, 3, 224, 224]
from datasets import load_dataset
#ds = load_dataset("SimulaMet-HOST/Kvasir-VQA")
ds_train = load_dataset("SimulaMet/Kvasir-VQA-x1", split="train")
ds_test = load_dataset("SimulaMet/Kvasir-VQA-x1", split="test")
from PIL import Image
import requests
import torch
import torchvision.transforms as transforms
transformten = transforms.Compose([
        transforms.Resize((224, 224)),   # adjust size for your model
        transforms.ToTensor(),           # convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])



def preprocess_example(example):
    # Download image
    image = Image.open(requests.get(example["image"], stream=True).raw).convert("RGB")
    
    # Apply your normalize/transform method
    image = transformten(image)  # e.g. Resize + ToTensor + Normalize


    #print("DEBUG image:", type(image), image.shape)

    # Tokenize the question
    q_inputs = router_tokenizer(example["question"], 
                                return_tensors="pt", 
                                truncation=True, 
                                padding="max_length", 
                                max_length=32)

    # q_inputs is a BatchEncoding with tensors inside (batch_size=1), so we squeeze
    input_ids = q_inputs["input_ids"].squeeze(0)          # torch.Tensor [seq_len]
    attention_mask = q_inputs["attention_mask"].squeeze(0)
    
    # Pack features
    return {
        "image": image,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answer": example["answer"],
        "question_class": example["question_class"],
        "image_url": example["image"],
    }


split_dataset_dict = ds_train.train_test_split(test_size=0.2, seed=42)
if(samples>=0):
    train_dataset = split_dataset_dict['train'].select(range(samples))
    val_dataset = split_dataset_dict['test'].select(range(samples))
else:
    train_dataset = split_dataset_dict['train']
    val_dataset = split_dataset_dict['test']


#print(train_dataset)
#print(val_dataset)

train_data = train_dataset.map(preprocess_example)
val_data = val_dataset.map(preprocess_example)

#print(train_data[0].keys())
#print(type(train_data[0]['image']))

# Tell HF to keep tensors
#train_data.set_format(type="torch", columns=["image"])
#val_data.set_format(type="torch", columns=["image"])


#print(train_data[0].keys())
#print(type(train_data[0]['image']))