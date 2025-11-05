# ==============================================
# flan_t5_classifier.py
# ==============================================
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class FLANT5Classifier(nn.Module):
    def __init__(self, model_name="google/flan-t5-base"):
        super(FLANT5Classifier, self).__init__()
        # Load pretrained FLAN-T5 model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def forward(self, input_texts, max_length=64):
        # Tokenize input text
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Generate output sequence
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length
        )

        # Decode the generated tokens
        decoded_outputs = [
            self.tokenizer.decode(g, skip_special_tokens=True) for g in outputs
        ]
        return decoded_outputs


if __name__ == "__main__":
    # Example usage
    model = FLANT5Classifier("google/flan-t5-small")  # smaller version for testing

    # Example input
    input_texts = [
        "Classify this call: 'Please transfer your bank details immediately!'",
        "Classify this call: 'Hello, how are you today?'"
    ]

    # Forward pass
    results = model(input_texts)
    for r in results:
        print("Generated:", r)
