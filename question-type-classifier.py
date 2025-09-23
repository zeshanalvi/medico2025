import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModel, AutoProcessor, VisionEncoderDecoderModel,
    T5Tokenizer, T5ForConditionalGeneration
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

router_name = "distilbert-base-uncased"
router_tokenizer = AutoTokenizer.from_pretrained(router_name)
router_model = AutoModelForSequenceClassification.from_pretrained(
    router_name, num_labels=6  # Yes/No, Single, Multi, Color, Location, Count
).to(device)

def classify_question_type(question):
    inputs = router_tokenizer(question, return_tensors="pt", truncation=True).to(device)
    outputs = router_model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return pred

class QuestionTypeClassifier(nn.Module):
    def __init__(self, hidden=768, num_types=len(q_types)):
        super().__init__()
        self.fc = nn.Linear(hidden, num_types)

    def forward(self, q_feat):
        return self.fc(q_feat)  # logits [B, 6]
        
qtype_classifier=QuestionTypeClassifier().to(device)


