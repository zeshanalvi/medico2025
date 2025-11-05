
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
from collections import defaultdict
from torch.utils.data import DataLoader
import os
from transformers import AutoTokenizer

image_cache = {}

def preprocess_image(image_source):
    """
    Preprocess a single image for inference.
    `image_source` can be either a URL or a local file path.
    Returns a tensor [C, H, W].
    """
    if isinstance(image_source, str):
        if image_source.startswith("http"):  # URL
            image = Image.open(requests.get(image_source, stream=True).raw).convert("RGB")
        else:  # local path
            image = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, Image.Image):  # already a PIL image
        image = image_source
    else:
        raise ValueError("Unsupported image_source type")

    # Apply the same transform used during training
    image = transformten(image)  # e.g. Resize(224) → ToTensor() → Normalize()

    return image  # torch.Tensor [3, H, W]
    
def preprocess_example(example,data_path):
    # Download image
    #image = Image.open(requests.get(example["image"], stream=True).raw).convert("RGB")

    router_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased",legacy=False)# New Behaviour

    #Image from dataset
    image_name = example["image"].split("/")[-1]
    image_path = os.path.join(data_path, image_name)

    # 2. Check if the image is already in our cache
    if image_path in image_cache:
        image = image_cache[image_path]
        
    else:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_cache[image_path] = image  # Cache the loaded image object
    
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
        "question": example["question"],
        "question_class": example["question_class"],
        "image_url": example["image"],
    }

def normalize_answer(ans, q_type):
    ans = ans.strip().lower()

    if q_type == "yesno":
        if "yes" in ans or "present" in ans or "evidence" in ans:
            return "Yes"
        elif "no" in ans or "absent" in ans or "none" in ans:
            return "No"
        else:
            return None  # ambiguous

    if q_type == "count":
        # Extract numeric value or return None
        from re import findall
        numbers = findall(r"\d+", ans)
        if numbers:
            return numbers[0]
        elif "one" in ans: return "1"
        elif "two" in ans: return "2"
        return None

    if q_type == "color":
        for color in ["red","green","yellow","blue","white","black"]:
            if color in ans:
                return color
        return None

    if q_type == "location":
        # Simplify locations to a small fixed set
        for loc in ["upper","lower","left","right","central"]:
            if loc in ans:
                return loc
        return None

    if q_type in ["single","multi"]:
        return ans  # keep original but can also restrict choices

    return ans


def build_vocabs(dataset,q_types_mapping):
    # Build task-specific vocabularies
    task_vocabs = {}
    for general_class in set(q_types_mapping.values()):
        task_vocabs[general_class] = {}
    
    for row in dataset:
        fine_class = row["question_class"]

        # ✅ Handle if fine_class is a list
        if isinstance(fine_class, list):
            fine_class = fine_class[0]  

        general_class = q_types_mapping[fine_class]

        norm_ans = normalize_answer(row["answer"], general_class)
        if norm_ans is None:
            continue  # skip unnormalizable answers

        if norm_ans not in task_vocabs[general_class]:
            idx = len(task_vocabs[general_class])
            task_vocabs[general_class][norm_ans] = idx

    return task_vocabs


def build_answer_vocab(dataset, q_types_mapping):
    answer_vocab = defaultdict(dict)
    counters = defaultdict(int)

    for ans, q_class in zip(dataset["answer"], dataset["question_class"]):
        # q_class might be a list; pick the first (if multiple labels)
        if isinstance(q_class, list):
            q_class = q_class[0]

        general_class = q_types_mapping[q_class]

        if ans not in answer_vocab[general_class]:
            answer_vocab[general_class][ans] = counters[general_class]
            counters[general_class] += 1

    return answer_vocab



def collate_fn(batch):
    #print(type(batch[0]["image"]))
    
    #images = torch.stack([item["image"] for item in batch])
    images = torch.stack([torch.tensor(item["image"]) if isinstance(item["image"], list) else item["image"] for item in batch])
    
    #print(type(images), images.shape)


    input_ids = torch.stack([torch.tensor(item["input_ids"]) if isinstance(item["input_ids"], list) else item["input_ids"] for item in batch])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"]) if isinstance(item["attention_mask"], list) else item["attention_mask"] for item in batch])


    
    #input_ids = torch.stack([item["input_ids"] for item in batch])
    #attention_mask = torch.stack([item["attention_mask"] for item in batch])
    answers = [item["answer"] for item in batch]  # keep as list for label encoding later
    question = [item["question"] for item in batch]  # keep as list for label encoding later
    q_classes = [item["question_class"] for item in batch]
    return {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answers": answers,
        "question": question,
        "question_classes": q_classes,
    }

