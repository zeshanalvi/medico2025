import torch
import torch.nn as nn
import sys, os
from git import Repo
import os
import numpy as np

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModel, AutoProcessor, VisionEncoderDecoderModel,
    T5Tokenizer, T5ForConditionalGeneration
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen_name = "t5-base"
gen_tokenizer = T5Tokenizer.from_pretrained(gen_name)
gen_model = T5ForConditionalGeneration.from_pretrained(gen_name).to(device)

def generate_descriptive_answer(question, prediction, fused_features):
    # Construct a prompt combining prediction and context
    prompt = f"Question: {question} | Prediction: {prediction} | Context: GI disease analysis"
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = gen_model.generate(**inputs, max_length=50)
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)



def clone_repo():
    repo_url = "https://github.com/zeshanalvi/Feature-Extraction.git"
    destination = "./Feature-Extraction"
    Repo.clone_from(repo_url, destination)
    print("Repository cloned successfully!")

def clone_repo():
    repo_url = "https://github.com/zeshanalvi/Feature-Extraction.git"
    destination = "./Feature-Extraction"
    
    # Skip if repo already exists
    if os.path.exists(destination):
        #print(f"Repository already exists at {destination}. Skipping clone.")
        return
    print("Cloning repository...")
    Repo.clone_from(repo_url, destination)
    print("Repository cloned successfully.")

def diseasem(model,img):
    clone_repo()
    from features import extract_features_batch
    repo_path = os.path.abspath("Feature-Extraction")
    if repo_path not in sys.path:
        sys.path.append(repo_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res=extract_features_batch(img)
    pred=model.predict_prob(res)
    disease_vec=torch.tensor(pred, dtype=torch.float32).to(device)
    return disease_vec

def disease_model(img):
    return torch.tensor(np.random.rand(23)).to(device)

router_name = "distilbert-base-uncased"
router_tokenizer = AutoTokenizer.from_pretrained(router_name)