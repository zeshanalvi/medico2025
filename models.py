import torch
import torch.nn as nn

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

def disease_model(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #torch.tensor(np.random.rand(23)).to(device)
    return torch.zeros(23).to(device)

router_name = "distilbert-base-uncased"
router_tokenizer = AutoTokenizer.from_pretrained(router_name)

