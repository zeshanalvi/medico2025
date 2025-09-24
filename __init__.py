"""
from qtype import QuestionTypeClassifier
from tpred import TaskPredictor
from model import VQAModel
from fussionmodel import CoAttentionFusion
from functions import preprocess_example, preprocess_image, collate_fn
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
"""

from .functions import preprocess_example, preprocess_image, collate_fn
from .model import VQAModel

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    samples=10
    print("Dataset Loading...")

    ds_train = load_dataset("SimulaMet/Kvasir-VQA-x1", split="train")
    ds_test = load_dataset("SimulaMet/Kvasir-VQA-x1", split="test")
    print("Dataset Loaded")

    split_dataset_dict = ds_train.train_test_split(test_size=0.2, seed=42)
    if(samples>0):
        train_dataset = split_dataset_dict['train'].select(range(samples))
        val_dataset = split_dataset_dict['test'].select(range(samples))
    else:
        train_dataset = split_dataset_dict['train']
        val_dataset = split_dataset_dict['test']

    print("Loading Images...")
    
    train_data = train_dataset.map(preprocess_example)
    val_data = val_dataset.map(preprocess_example)
    print("Images Loaded")

    
    print("Making Data Loader...")

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=collate_fn)

    print("Data Loader Completed")




    a=QuestionTypeClassifier(num_types=1)
    tp=TaskPredictor(task_type = "yesno")
    cf=CoAttentionFusion()
    model = VQAModel(img_dim=768, ques_dim=768, disease_dim=23, hidden_dim=512).to(device)
    model.train(epochs=1,data_train=train_data,train_loader=train_loader)
    model.save()
    res=model.evaluate(val_loader)