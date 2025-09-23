#!git clone https://github.com/zeshanalvi/feature-extraction.git
import sys
#sys.path.append("/kaggle/working/feature-extraction")
#from features import get_lires, get_lbps, Dataset, color_layout
import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from datasets import load_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_hfimage(img_path):
    response = requests.get(img_path, stream=True)
    if response.status_code == 200:
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img
def get_features(img):
    cl,t1=color_layout(img)
    return cl
def disease_model(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(np.random.rand(23)).to(device)
def get_class_all(dataset):
    tensors = []
    for d in dataset:
        tensors.append(disease_model(d['image']))
    return torch.cat(tensors, dim=0)
        
#disease_probs=get_class_all(dataset)