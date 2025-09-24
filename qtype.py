import torch
from torch import nn
from transformers import DistilBertModel

class QuestionTypeClassifier(nn.Module):
    def __init__(self, num_types):
        super().__init__()
        
        # Load pre-trained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        # Classification head
        self.fc = nn.Linear(self.distilbert.config.hidden_size, num_types)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Take [CLS] token embedding (DistilBERT uses first token as [CLS])
        cls_token = outputs.last_hidden_state[:, 0, :]  # [B, hidden]
        
        logits = self.fc(cls_token)  # [B, num_types]
        return logits
