from .model import VQAModel
import torch, gc
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from treenet import TreeNet
import argparse

# Create parser
parser = argparse.ArgumentParser(description="Example of labeled arguments")

# Define arguments
parser.add_argument("-e","--epochs", type=int, default=1, help="Number of Epochs")
parser.add_argument("-s","--batch", type=int, default=16, help="Batch size")
parser.add_argument("-b","--batches", type=int, default=0, help="Batches if set as zero then whole data will be processed")
parser.add_argument("-f","--finetune", type=int, default=0, help="Finetune-1 or Retrain-0")
parser.add_argument("-w","--weights", type=str, default="na", help="Weights File Path")
parser.add_argument("-bs","--startbatch", type=int, default=0, help="Start from Batch")

# Parse arguments
args = parser.parse_args()

batch_size=args.batch
batches=args.batches
epochs=args.epochs
finetune=False
if(args.finetune==1):
    finetune=True
weights_path=args.weights
st_batch=args.startbatch

from .functions import preprocess_example, collate_fn

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print("Dataset Loading...")

ds_train = load_dataset("SimulaMet/Kvasir-VQA-x1", split="train")
ds_test = load_dataset("SimulaMet/Kvasir-VQA-x1", split="test")
print("Dataset Loaded")

disease_model = TreeNet(layer_count=6, breath_count=3)
import os
weights_treenet="/home/dr-zeeshan-khan/Desktop/code/Medico2025/treenet_model63.pkl" #Path to the treenet weights
base_path="/home/dr-zeeshan-khan/Desktop/datasets/me2025/images/"
if os.name == "nt":
    print("Running on Windows")
    weights_treenet="C:\\Users\\Zeeshan\\Documents\\GitHub\\Medico2025\\treenet_model63.pkl"
    base_path="D:\\datasets\\mediaEval2025\\me2025\\images\\"
elif os.name == "posix":
    print("Running on Linux or macOS")
    weights_treenet="/home/dr-zeeshan-khan/Desktop/code/Medico2025/treenet_model63.pkl"
    base_path="/home/dr-zeeshan-khan/Desktop/datasets/me2025/images/"


disease_model.load(weights_treenet)


if(batches==0):
    batches=int(ds_train.num_rows/batch_size)
print("Total Batches\t",batches)



for b in range(st_batch,batches):
    print("Processing Batch \t",b+1)
    batch=ds_train.select(range(b*batch_size,min((b+1)*batch_size,ds_train.num_rows)))
    split_dataset_dict = batch.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset_dict['train']
    val_dataset = split_dataset_dict['test']

    print("Loading Images...")    
    train_data = train_dataset.map(preprocess_example,fn_kwargs={"data_path": base_path})
    #print(train_data["input_ids"])
    val_data = val_dataset.map(preprocess_example,fn_kwargs={"data_path": base_path})
    print("Images Loaded")


    print("Making Data Loader...")
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    print("Data Loader Completed")

    #print(train_data)

    # 🧹 Force cleanup before starting new batch
    gc.collect()
    torch.cuda.empty_cache()

    # 🚫 Delete old model and optimizer to free VRAM
    if 'model' in locals():
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 🧠 Optionally synchronize GPU
    elif os.name == "posix":
        torch.cuda.synchronize()

    try:
        model = VQAModel(img_dim=768, ques_dim=768, disease_dim=23, hidden_dim=512)#.to(device)
        print("\n\nModel created successfully.")
    except Exception as e:
        print("\n\nError during model creation:", e)
    
    print("Model Training...")
    
    if(b==0 and finetune==False):
        model.train_model(epochs=epochs,data_train=train_data,train_loader=train_loader,disease_model=disease_model)
        model.save("vqan_"+str(b%10)+".pt")
        print("Batch ",str(b+1)," Completed")
    else:
        if(finetune==True):
            model.load("vqan_"+str(b%10)+".pt")
        else:
            model.load(weights_path)
        #model.fine_tune_model(train_loader=train_loader,val_loader=val_loader,epochs=1,unfreeze_encoders=False,save_best_path=None,validate_every=1)
        #model.fine_tune_model(epochs=epochs, data_train=train_data, train_loader=train_loader, disease_model=disease_model, checkpoint_path="vqan_"+str((b)%10)+".pt")
        model.fine_tune_model1(train_loader=train_loader,
                               data_train=train_data,
                               val_loader=val_loader,
                               epochs=epochs,
                               unfreeze_encoders=False,
                               save_best_path=None,
                               disease_model=disease_model,
                               validate_every=1)
    model.save("vqan_"+str((b+1)%10)+".pt")