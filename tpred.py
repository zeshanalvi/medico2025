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
    BlipForConditionalGeneration,
    AutoModel,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BitsAndBytesConfig,
    AutoConfig
)
from .ImageToTextProjector import ImageToTextProjector


class TaskPredictor(nn.Module):
    def __init__(self, task_type, hidden=512, device='cuda'):
        super().__init__()
        self.task_type = task_type
        
        #print("Should be any\t",self.task_type)
        #for testing
        #print("Task Type\t",self.task_type)
        #if self.task_type not in ["yesno", "single", "location", "count"]:
        #    self.task_type = "yesno"

        self.device = device

        # Flags to track lazy loading
        self._loaded = False

        self.projector = ImageToTextProjector().to(self.device)

        self._lazy_load()

    def _lazy_load(self):
        if self._loaded:
            return  
        if self.task_type in ["yesno", "single"]:#, "color"]:
            model_name = "google/flan-t5-small" # 1 GB
            self.tokenizer = AutoTokenizer.from_pretrained(model_name,low_cpu_mem_usage=True)
            self.head = AutoModelForSeq2SeqLM.from_pretrained(model_name,low_cpu_mem_usage=True).to(self.device)

        elif self.task_type == "single":
            model_name = "facebook/bart-base" # 2 GB
            #self.tokenizer = AutoTokenizer.from_pretrained(model_name,low_cpu_mem_usage=True)
            #self.head = AutoModelForSeq2SeqLM.from_pretrained(model_name,low_cpu_mem_usage=True).to(self.device)

        elif self.task_type == "multi":
            #model_name = "microsoft/deberta-v3-xsmall" # 1 GB
            model_name = "MoritzLaurer/DeBERTa-v3-base-mnli"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name,low_cpu_mem_usage=True)
            # Load config and adjust num_labels for your downstream task
            config = AutoConfig.from_pretrained(model_name)
            config.num_labels = 35  # <-- number of possible answer choices in your task
            # Initialize model with updated config
            self.head = AutoModelForMultipleChoice.from_pretrained(model_name,config=config,ignore_mismatched_sizes=True,low_cpu_mem_usage=True)
            # Force materialization of meta tensors
            self.head.to_empty(device=self.device)
            self.head.load_state_dict(self.head.state_dict())  # ensures data tensors
            self.head.to(self.device)
            #self.head = AutoModelForMultipleChoice.from_pretrained(model_name,low_cpu_mem_usage=True).to(self.device)# Warning multiple choice and single
            self.head.gradient_checkpointing_enable()

        elif self.task_type == "color":
            model_name = "Salesforce/blip-image-captioning-base" # 3 GB
            #model_name = "google/flan-t5-small" # 1 GB
            #model_name = "google/flan-t5-small"  # 1 GB, lighter than BLIP
            #self.tokenizer = AutoTokenizer.from_pretrained(model_name, low_cpu_mem_usage=True, legacy=False)
            #self.head = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(self.device)            
            self.processor = AutoProcessor.from_pretrained(model_name, low_cpu_mem_usage=True)
            self.head = BlipForConditionalGeneration.from_pretrained(model_name, low_cpu_mem_usage=True).to(self.device)

        elif self.task_type == "location":
            # Configure 8-bit quantization
            #quant_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)
            #model_name = "Salesforce/blip2-flan-t5-xl" # 8 GB
            #self.processor = AutoProcessor.from_pretrained(model_name,low_cpu_mem_usage=True)
            #self.head = Blip2ForConditionalGeneration.from_pretrained(model_name,quantization_config=quant_config,low_cpu_mem_usage=True,device_map="auto")#.to(self.device)
            
            model_name = "google/flan-t5-small" # 1 GB
            self.tokenizer = AutoTokenizer.from_pretrained(model_name,low_cpu_mem_usage=True)
            self.head = AutoModelForSeq2SeqLM.from_pretrained(model_name,low_cpu_mem_usage=True).to(self.device)

        elif self.task_type == "count":
            #model_name = "microsoft/git-base" # 4 GB
            #model_name = "t5-base" # 4 GB            
            model_name = "google/flan-t5-small" # 1 GB
            self.tokenizer = AutoTokenizer.from_pretrained(model_name,low_cpu_mem_usage=True)
            self.head = AutoModelForSeq2SeqLM.from_pretrained(model_name,low_cpu_mem_usage=True).to(self.device)

        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        self._loaded = True

    def forward(self, x, question,raw_image=None, **kwargs):
        # Load model on-demand
        self._lazy_load()

        #print("Data Type of the input",type(x),"with shape",x.shape)
        #print("Question submitted\t",question)
        #print("task_type",self.task_type)
        #print("question_embedings",self.question_embedings(question,x))

        #if self.device == "cuda":# May get errors on GPU machine
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        #print("Predicting Answers...", self.task_type)
        with torch.no_grad():
            projected = self.projector(x)
        image_text = " ".join([f"{v:.3f}" for v in projected[0].tolist()])
        prompt = f"Question: {question}\nImage features: {image_text}\nAnswer:"

        # ---------- YES/NO, SINGLE, LOCATION, COUNT ----------
        if self.task_type in ["yesno", "single", "location", "count"]:#, "color"]:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, padding=True, max_length=128
            ).to(self.device)
            outputs = self.head(**inputs, labels=inputs["input_ids"])
            #print("Logits:", outputs.logits.shape)
            # Shape: (1, sequence_length, vocab_size)
            return outputs.logits

        # ---------- MULTI-CHOICE ----------
        elif self.task_type == "multi":
            # Multiple choice expects multiple options
            # Example options
            choices=['text_presence','procedure_type','polyp_type','polyp_size','polyp_removal_status','polyp_count',
            'landmark_presence','landmark_location','landmark_color','instrument_presence','instrument_location',
            'instrument_count','finding_presence','finding_count','box_artifact_presence','abnormality_presence',
            'abnormality_location','abnormality_color','text_presence','procedure_type','polyp_type','polyp_size',
            'polyp_removal_status','polyp_count','landmark_presence','landmark_location','landmark_color',
            'instrument_presence','instrument_location','instrument_count','finding_count','box_artifact_presence',
            'abnormality_presence','abnormality_location','abnormality_color']

            #choices = ["option A", "option B", "option C", "option D"]

            # Encode prompt + each choice
            encoding = self.tokenizer(
                [f"{prompt} {choice}" for choice in choices],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)

            outputs = self.head(**{k: v.unsqueeze(0) for k, v in encoding.items()})
            #print("Logits:", outputs.logits.shape)
            # Shape: (1, num_choices)
            pred = torch.argmax(outputs.logits, dim=1)
            return pred.item(), outputs.logits

        # ---------- COLOR ----------
        elif self.task_type == "color1":
            # x should be actual images, e.g., [B, 3, H, W] tensors or PIL images
            prompt = f"Question: {question}"
            inputs = self.processor(images=x, text=prompt, return_tensors="pt").to(self.device)
            outputs = self.head.generate(**inputs, max_new_tokens=10)
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            return answer
            # Use T5 instead of BLIP
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
            outputs = self.head(**inputs, labels=inputs["input_ids"])
            answer_logits = outputs.logits  # shape: [1, seq_len, vocab_size]

            return answer_logits
        
            # For BLIP models: pass image and text prompt
            inputs = self.processor(images=x, text=prompt, return_tensors="pt").to(self.device)
            outputs = self.head.generate(**inputs, max_new_tokens=10)
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            print("Generated caption/color:", answer)
            return answer
        
        elif self.task_type == "color":
            import PIL
            # Ensure PIL image (your image already is)
            assert isinstance(raw_image, PIL.Image.Image)

            # BLIP prompt must be simple
            prompt = question.strip()

            # Processor: encode image + text
            inputs = self.processor(
                images=raw_image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            # Generate output
            outputs = self.head.generate(
                **inputs,
                max_new_tokens=10
            )

            # Decode caption answer
            answer = self.processor.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            ).strip().lower()

            #print("BLIP color answer:", answer)
            return answer
        
        elif self.task_type == "color2":
            # raw_question is an image tensor shaped [1, 3, H, W]
            print(type(raw_image),raw_image)
            prompt = f"Question: {question}"

            inputs = self.processor(
                images=raw_image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.head.generate(
                **inputs,
                max_new_tokens=10
            )

            # Correct decoding
            answer = self.processor.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            ).strip().lower()

            print("answer for color",answer)

            return answer

        else:
            #print("Predicting Answers...",self.task_type)
            projector = ImageToTextProjector().to(self.device)
            with torch.no_grad():
                   projected = projector(x)
            image_text = " ".join([f"{v:.3f}" for v in projected[0].tolist()])
            prompt = f"Question: {question}\nImage features: {image_text}\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
            outputs = self.head.generate(**inputs, max_length=50)
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            #print("Question\t",question,"\nPredicted Answer\t", answer)
            #print(self.task_type,"\t",answer)
            return answer
            #return self.head(x)
    
    def question_embedings(self,questions,image_vector):
        
        # ------------------------------------------------------------
        # 2. Prepare inputs
        # ------------------------------------------------------------
        print(questions)
        question = "What is the condition of the patient based on the image?"

        # Tokenize the question
        inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(self.device)

        #for k, v in inputs.items():
        #    print(k, v.device)
        #    print("Model device:", next(self.text_model.parameters()).device)

        # Get the question embeddings (encoder output)
        with torch.no_grad():
            outputs = self.text_model.encoder(**inputs).to(self.device)
            question_embedding = outputs.last_hidden_state  # [1, seq_len, hidden_dim]

        print("Question embedding:", question_embedding.shape)
        # Example: [1, 17, 512]

        # ------------------------------------------------------------
        # 3. Example image vector (from some vision encoder, e.g., ViT)
        # ------------------------------------------------------------
        #image_vector = torch.randn(1, 512)  # replace with your actual image features

        print("Image vector:", image_vector.shape)
        # [1, 512]

        # ------------------------------------------------------------
        # 4. Combine (simple mean pooling over question tokens)
        # ------------------------------------------------------------
        question_pooled = question_embedding.mean(dim=1)  # [1, 512]

        # Concatenate both
        joint_vector = torch.cat([image_vector, question_pooled], dim=1)  # [1, 1024]
        print("Joint vector shape:", joint_vector.shape)
        return joint_vector
    
