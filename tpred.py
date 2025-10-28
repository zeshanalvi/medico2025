# ---------------------------
# Step 5: Task-Specific Predictors
# ---------------------------
import torch.nn as nn
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


class TaskPredictor1(nn.Module):
    def __init__(self, task_type, hidden=512):
        super().__init__()
        if task_type == "yesno":
            #self.head = nn.Linear(hidden, 2)
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            self.head = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        elif task_type == "single":
            #self.head = nn.Linear(hidden, 10)
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
            self.head = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
        elif task_type == "multi":
            #self.head = nn.Linear(hidden, 10)
            #self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
            #self.head = AutoModelForMultipleChoice.from_pretrained("microsoft/deberta-v3-base")
            #Lighter versions
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")
            self.head = AutoModelForMultipleChoice.from_pretrained("microsoft/deberta-v3-xsmall")
            self.head.gradient_checkpointing_enable()
        elif task_type == "color":
            #self.head = nn.Linear(hidden, 5)
            #self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            #self.head = AutoModelForVision2Seq.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            #Lighter Version
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.head = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            # Color-related reasoning
            #self.processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-small")
            #self.head = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-small")

        elif task_type == "location":
            #self.head = nn.Linear(hidden, 6)
            #self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
            #self.head = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
            #Lighter Version
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.head = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
            #self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-small")
            #self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-small")

        elif task_type == "count":
            #self.head = nn.Linear(hidden, 1)
            #self.tokenizer = AutoTokenizer.from_pretrained("t5-large")
            #self.head = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
            #Lighter Version
            self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
            self.head = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        else:
            raise ValueError("Unknown task")
    
    def forward(self, x):
        if self.task_type == "yesno":
            # Expect x as text input(s)
            inputs = self.tokenizer(
                x, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            outputs = self.head.generate(
                **inputs,
                max_length=5,
                num_beams=2
            )
            decoded = [
                self.tokenizer.decode(o, skip_special_tokens=True).strip().lower()
                for o in outputs
            ]
            return decoded
        elif self.task_type == "single":
            # Expect x to be a list of text prompts for BART
            inputs = self.tokenizer(
                x,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            outputs = self.head.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=64
            )
            decoded = [self.tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
            return decoded
        elif self.task_type == "multi":
            # Multiple-choice using DeBERTa-v3
            # Expect input as a list of (question, choices)
            questions = [q for q, _ in x]
            choices = [opts for _, opts in x]

            encoded = self.tokenizer(
                [[q + " " + c for c in choice] for q, choice in x],
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            outputs = self.head(**encoded)
            return outputs.logits  # (batch, num_choices)

        elif self.task_type == "color":
            # Vision-language using ViT–GPT2
            images = x  # Expect preprocessed image tensors
            outputs = self.head.generate(**kwargs)
            captions = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            return captions
        # --- Location-based (BLIP2–FLAN–T5–XL) ---
        elif self.task_type == "location":
            # Expect image + text in kwargs
            inputs = self.processor(**kwargs, return_tensors="pt")
            outputs = self.head.generate(**inputs, max_length=64)
            return [self.processor.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

        # --- Counting (T5-Large) ---
        elif self.task_type == "count":
            inputs = self.tokenizer(
                x, return_tensors="pt", padding=True, truncation=True
            )
            outputs = self.head.generate(**inputs, max_length=10)
            return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        else:
            # For all other tasks, x is assumed to be a tensor
            return self.head(x)

    def forward1(self, x):
        return self.head(x)
    
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

        # Placeholders for models/tokenizers
        self.tokenizer = None
        self.processor = None
        self.head = None

        # Flags to track lazy loading
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return  # Already loaded

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
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-small")
            self.head = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-small").to(self.device)

        elif self.task_type == "count":
            self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
            self.head = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(self.device)

        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        self._loaded = True

    def forward(self, x, **kwargs):
        # Load model on-demand
        self._lazy_load()
        
        if isinstance(x, torch.Tensor):
            return self.head(x)
        
        if self.task_type == "yesno":
            inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.head.generate(**inputs, max_length=5, num_beams=2)
            return [self.tokenizer.decode(o, skip_special_tokens=True).strip().lower() for o in outputs]
        
        elif self.task_type == "single":
            inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.head.generate(**inputs, max_length=64)
            return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

        #if self.task_type == "yesno":
        #    inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True).to(self.device)
        #    outputs = self.head.generate(**inputs, max_length=5, num_beams=2)
        #    return [self.tokenizer.decode(o, skip_special_tokens=True).strip().lower() for o in outputs]

        #elif self.task_type == "single":
        #    inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True).to(self.device)
        #    outputs = self.head.generate(**inputs, max_length=64)
        #    return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

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

        elif self.task_type == "count":
            inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.head.generate(**inputs, max_length=10)
            return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

        else:
            return self.head(x)
