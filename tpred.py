# ---------------------------
# Step 5: Lazy-Load Task-Specific Predictors
# ---------------------------
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForMultipleChoice, 
    AutoModelForVision2Seq, 
    Blip2ForConditionalGeneration, 
    AutoProcessor, 
    BlipForConditionalGeneration
)

class TaskPredictor(nn.Module):
    def __init__(self, task_type, hidden=512, device='cuda'):
        super().__init__()
        self.task_type = task_type
        self.device = device
        self.tokenizer = None
        self.processor = None
        self.head = None
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return

        if self.task_type == "yesno":
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            self.head = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(self.device)
        elif self.task_type == "single":
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
            self.head = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base").to(self.device)
        elif self.task_type == "multi":
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")
            self.head = AutoModelForMultipleChoice.from_pretrained("microsoft/deberta-v3-xsmall").to(self.device)
            self.head.gradient_checkpointing_enable()
        elif self.task_type == "color":
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.head = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        elif self.task_type == "location":
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.head = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(self.device)
        elif self.task_type == "count":
            self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
            self.head = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(self.device)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        self._loaded = True
    
    def forward(self, x, **kwargs):
        self._lazy_load()

        # --- If input is tensor, just forward it ---
        if isinstance(x, torch.Tensor):
            return self.head(x.to(self.device))

        # --- Otherwise, assume text input ---
        if self.task_type in ["yesno", "single", "count"]:
            if isinstance(x, str):
                x = [x]  # single string → list
            inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.head.generate(**inputs, max_length=64)
            return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

        elif self.task_type == "multi":
            # x = list of (question, choices)
            encoded = self.tokenizer(
                [[q + " " + c for c in choice] for q, choice in x],
                return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            outputs = self.head(**encoded)
            return outputs.logits

        elif self.task_type == "color":
            images = x
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.head.generate(**inputs)
            return [self.processor.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

        elif self.task_type == "location":
            inputs = self.processor(**kwargs, return_tensors="pt").to(self.device)
            outputs = self.head.generate(**inputs, max_length=64)
            return [self.processor.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


    def forward(self, x, **kwargs):
        self._lazy_load()

        # --- Dual-mode forward ---
        if isinstance(x, torch.Tensor):
            # Input is a fused embedding tensor → use tensor head
            return self.head(x.to(self.device))

        # --- Text input mode ---
        if self.task_type in ["yesno", "single", "count"]:
            inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.head.generate(**inputs, max_length=64)
            return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

        elif self.task_type == "multi":
            encoded = self.tokenizer([[q + " " + c for c in choice] for q, choice in x],
                                     return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.head(**encoded)
            return outputs.logits

        elif self.task_type == "color":
            inputs = self.processor(images=x, return_tensors="pt").to(self.device)
            outputs = self.head.generate(**inputs)
            return [self.processor.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

        elif self.task_type == "location":
            inputs = self.processor(**kwargs, return_tensors="pt").to(self.device)
            outputs = self.head.generate(**inputs, max_length=64)
            return [self.processor.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        
        else:
            return self.head(x)
