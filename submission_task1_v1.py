import torch
import torch.nn as nn
from datasets import load_dataset, Image as HfImage
from transformers import AutoProcessor, AutoTokenizer
import json, time, platform, sys, subprocess
from tqdm import tqdm
from evaluate import load

# ================== METRICS ================== #
bleu = load("bleu")
rouge = load("rouge")
meteor = load("meteor")

# ================== DATASET ================== #
ds = load_dataset("SimulaMet/Kvasir-VQA-x1")["test"]
ds_shuffled = ds.shuffle(seed=42)
val_dataset = ds_shuffled.select(range(1500))
val_dataset = val_dataset.cast_column("image", HfImage())

predictions = []
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_mem():
    return torch.cuda.memory_allocated(device)/(1024**2) if torch.cuda.is_available() else 0

initial_mem = get_mem()

# ================== SUBMISSION INFO ================== #
SUBMISSION_INFO = {
    "Participant_Names": "Zeshan Khan",
    "Affiliations": "National University of Computer and Emerging Sciences",
    "Contact_emails": ["zeshankhanalvi@gmail.com"],
    "Team_Name": "FAST-NU-DS",
    "Country": "Pakistan",
    "Notes_to_organizers": "Custom pipeline with disease classifier + co-attention fusion."
}

# ================== IMPORT YOUR MODEL ================== #
from model import DiseaseClassifier, CoAttentionFusion, AnswerGenerator

# load pretrained disease classifier (you must save this separately or integrate HF repo)
disease_model = DiseaseClassifier().to(device)
disease_model.load_state_dict(torch.load("disease_classifier.pt", map_location=device))
disease_model.eval()

# co-attention fusion
fusion_model = CoAttentionFusion(img_dim=2048, ques_dim=768, disease_dim=23, hidden_dim=512).to(device)

# answer generator (choose LM decoder)
answer_generator = AnswerGenerator(num_classes=23).to(device)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ================== VALIDATION LOOP ================== #
start_time, post_model_mem = time.time(), get_mem()

for idx, ex in enumerate(tqdm(val_dataset, desc="Validating")):
    question = ex["question"]
    image = ex["image"].convert("RGB")

    # --- Step 1: Extract disease vector ---
    with torch.no_grad():
        dis_vec = disease_model(image).to(device)  # [23]

    # --- Step 2: Encode question ---
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(device)
    ques_feat = inputs["input_ids"]

    # --- Step 3: Get image features (CNN backbone placeholder) ---
    img_feat = torch.randn(1, 49, 2048).to(device)  # replace with real extractor (ResNet/ViT)

    # --- Step 4: Fusion ---
    fused = fusion_model(img_feat, ques_feat.mean(dim=1), dis_vec.unsqueeze(0))

    # --- Step 5: Generate answer ---
    answer = answer_generator(fused)

    assert isinstance(answer, str), f"Generated answer at index {idx} is not a string"
    predictions.append({"index": idx, "img_id": ex["img_id"], "question": question, "answer": answer})

# ================== METRICS ================== #
references = [[e] for e in val_dataset['answer']]
preds = [pred['answer'] for pred in predictions]

bleu_score = round(bleu.compute(predictions=preds, references=references)['bleu'], 4)
rouge_res = rouge.compute(predictions=preds, references=references)
meteor_score = round(meteor.compute(predictions=preds, references=references)['meteor'], 4)

public_scores = {
    'bleu': bleu_score,
    'rouge1': round(float(rouge_res['rouge1']), 4),
    'rouge2': round(float(rouge_res['rouge2']), 4),
    'rougeL': round(float(rouge_res['rougeL']), 4),
    'meteor': meteor_score
}
print("✨ Public scores: ", public_scores)

# ================== SAVE OUTPUT ================== #
output_data = {
    "submission_info": SUBMISSION_INFO,
    "public_scores": public_scores,
    "predictions": predictions,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
}
with open("predictions_1.json", "w") as f:
    json.dump(output_data, f, indent=4)
print("✅ Done. Results saved to predictions_1.json")
