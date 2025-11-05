import torch
import torch.nn as nn
import os
from .qtype import QuestionTypeClassifier
from .functions import build_vocabs, build_answer_vocab, collate_fn, preprocess_example, normalize_answer, preprocess_image
from .models import disease_model,diseasem, device, generate_descriptive_answer, router_tokenizer, gen_model
from .tpred import TaskPredictor
from .model_functions import compute_loss, compute_meteor, compute_rouge, extract_count, forward_batch
from .fussionmodel import BertModel, CoAttentionFusion, ViTModel, F
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModel

class VQAModel(nn.Module):
    def __init__(self,img_dim, ques_dim, disease_dim, hidden_dim):
        super(VQAModel, self).__init__()
        #self.fusion = CoAttentionFusion(img_dim, ques_dim, disease_dim, hidden_dim, answer_vocab=answer_vocab)
        self.epochs=1
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim=hidden_dim
        self.input_dim=768
        self.ques_dim=ques_dim
        self.disease_dim=disease_dim
        self.img_dim=img_dim
        self.fusion_module=None
        self.question_encoder=BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.image_encoder=ViTModel.from_pretrained("google/vit-base-patch16-224",add_pooling_layer=False,low_cpu_mem_usage=True).to(self.device)
        self.optimizer=None
        self.answer_vocabs=None
        self.task_vocabs=None
        self.data_train=None
        self.train_loader=None
        #self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small",low_cpu_mem_usage=True,legacy=False)
        self.text_model = AutoModel.from_pretrained("google/flan-t5-small")
        self.q_types = ["yesno", "single", "multi", "color", "location", "count"]
        self.qtype_classifier=QuestionTypeClassifier(num_types=len(self.q_types)).to(self.device)
        self.answer_classifier=nn.Linear(self.hidden_dim, 100)
        # Create task-specific heads (trainable)
        self.task_heads = nn.ModuleDict({
            t: TaskPredictor(t, hidden=hidden_dim,device=self.device) for t in self.q_types
        })
        self.q_types_mapping = {
            'abnormality_color': 'color',
            'landmark_color': 'color',
            'abnormality_location': 'location',
            'instrument_location': 'location',
            'landmark_location': 'location',
            'finding_count': 'count',
            'instrument_count': 'count',
            'polyp_count': 'count',
            'abnormality_presence': 'yesno',
            'box_artifact_presence': 'yesno',
            'finding_presence': 'yesno',
            'instrument_presence': 'yesno',
            'landmark_presence': 'yesno',
            'text_presence': 'yesno',
            'polyp_removal_status': 'yesno',
            'polyp_type': 'single',
            'polyp_size': 'single',
            'procedure_type': 'single',
        }
        
    
    def train_model(self,epochs,data_train,train_loader,disease_model):
        self.epochs=epochs
        self.train_data=data_train
        self.train_loader=train_loader
        self.answer_vocabs = build_answer_vocab(self.train_data, self.q_types_mapping)
        self.task_vocabs = build_vocabs(self.train_data,self.q_types_mapping)
        #self.qtype_classifier = nn.Linear(hidden_dim, len(self.task_vocabs))   # ✅ match hidden_dim
        self.qtype_classifier=QuestionTypeClassifier(num_types=len(self.q_types)).to(self.device)
        #QuestionTypeClassifier(hidden=self.input_dim, num_types=len(self.q_types)).to(device)
        #print(self.qtype_classifier)
        self.answer_classifier = nn.Linear(self.hidden_dim, len(self.answer_vocabs)) # ✅ match hidden_dim        
        self.fusion_module = CoAttentionFusion(img_dim=self.img_dim,
                                               ques_dim=self.ques_dim,
                                               disease_dim=self.disease_dim,
                                               hidden_dim=self.hidden_dim,
                                               answer_vocab=self.answer_vocabs).to(self.device)
        self.optimizer = torch.optim.AdamW(list(self.fusion_module.parameters()) + 
                                  list(self.question_encoder.parameters()) + 
                                  list(self.image_encoder.parameters())+
                                  list(self.qtype_classifier.parameters()), lr=2e-5)
        #print("Starting Training")
        #print("Total batches:", len(self.train_loader))
        for epoch in range(self.epochs):
            self.fusion_module.train()
            self.qtype_classifier.train()
            total_loss = 0
            #print("Device check:",next(self.image_encoder.parameters()).device,next(self.question_encoder.parameters()).device,next(self.fusion_module.parameters()).device)

            for batch in self.train_loader:
                #print("Data Types in Batch\t",type(batch["images"]),type(batch["input_ids"]),type(batch["attention_mask"]),type(batch["question"]),type(batch["answers"]),type(batch["question_classes"]),type(batch),)
                self.optimizer.zero_grad()
                #torch.cuda.empty_cache()
                #torch.cuda.ipc_collect()

                preds, answers, task_logits = forward_batch(
                    batch["images"],
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["question"],
                    batch["answers"],
                    batch["question_classes"],  # fine-grained from dataset
                    qtype_classifier=self.qtype_classifier,
                    fusion_module=self.fusion_module,
                    q_types=self.q_types,
                    q_types_mapping=self.q_types_mapping,
                    task_heads=self.task_heads,
                    device=self.device,
                    image_encoder=self.image_encoder,
                    question_encoder=self.question_encoder,
                    tokenizer=self.tokenizer,
                    disease_model=disease_model
                )
                #print("preds",preds)
                #print("answers",answers)
                #print("task_logits",task_logits)

                #preds, answers = forward_batch(batch["images"],batch["input_ids"], batch["attention_mask"], batch["answers"], batch["question_classes"])
                #print("\n\nCompute Loss\n preds\t",type(preds),"\n answers\t",type(answers),"\n task_logits\t",type(task_logits),"\n question_classes\t",type(batch["question_classes"]))
                #print("answer_vocabs\t",type(self.answer_vocabs),"\n q_types_mapping\t",type(self.q_types_mapping),"\n q_types\t",type(self.q_types),"\n task_heads\t",type(self.task_heads))
                #print("\nCompute Loss Dimensions\n","preds:",len(preds),"\t answers:",len(answers),"\t Logits:",task_logits.shape,"\t question_classes",len(batch["question_classes"]))
                #print("answer_vocabs:",len(self.answer_vocabs),"\t q_types_mapping:",len(self.q_types_mapping),"\tq_types",len(self.q_types),"\ttask_heads:",len(self.task_heads))
                #for t in self.task_heads:
                #    print(t)
                loss = compute_loss(preds,
                                    answers,
                                    task_logits,
                                    batch["question_classes"],
                                    answer_vocabs=self.answer_vocabs,
                                    q_types_mapping=self.q_types_mapping,
                                    q_types=self.q_types,
                                    task_heads=self.task_heads
                                   )
                #loss = compute_loss(preds, answers, batch["question_classes"])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}, Train Loss: {total_loss / len(train_loader)}")

    def fine_tune_model1(self, 
                    train_loader,
                    data_train=None,
                    val_loader=None, 
                    epochs=1,
                    unfreeze_encoders=False,
                    save_best_path=None,
                    disease_model=None,
                    validate_every=1):
        """
        Fine-tune the model on new data using previously trained weights.
        
        Args:
            train_loader: DataLoader for fine-tuning data
            val_loader: Optional validation DataLoader
            epochs: Number of fine-tuning epochs
            unfreeze_encoders: If False, keeps image & question encoders frozen
            save_best_path: Optional path to save best validation model
            validate_every: Perform validation after every N epochs
        """
        
        print("=== Starting Fine-tuning ===")
        print(f"Unfreeze Encoders: {unfreeze_encoders}")
        print(f"Fine-tuning for {epochs} epochs")

        # Optionally freeze or unfreeze encoders
        for param in self.image_encoder.parameters():
            param.requires_grad = unfreeze_encoders
        for param in self.question_encoder.parameters():
            param.requires_grad = unfreeze_encoders

        # Rebuild vocabularies if needed
        #print(self.answer_vocabs)
        self.answer_vocabs = build_answer_vocab(data_train, self.q_types_mapping)
        
        # Prepare optimizer (include all trainable params)
        trainable_params = (
            list(self.fusion_module.parameters()) +
            list(self.qtype_classifier.parameters()) +
            list(self.image_encoder.parameters()) +
            list(self.question_encoder.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            [p for p in trainable_params if p.requires_grad],
            lr=2e-5
        )

        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.fusion_module.train()
            self.qtype_classifier.train()
            total_loss = 0

            print(f"\n--- Fine-tune Epoch {epoch+1}/{epochs} ---")

            for batch in train_loader:
                self.optimizer.zero_grad()

                preds, answers, task_logits = forward_batch(
                    batch["images"],
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["question"],
                    batch["answers"],
                    batch["question_classes"],
                    qtype_classifier=self.qtype_classifier,
                    fusion_module=self.fusion_module,
                    q_types=self.q_types,
                    q_types_mapping=self.q_types_mapping,
                    task_heads=self.task_heads,
                    device=self.device,
                    image_encoder=self.image_encoder,
                    question_encoder=self.question_encoder,
                    tokenizer=self.tokenizer,
                    disease_model=disease_model
                )

                loss = compute_loss(preds,
                                    answers,
                                    task_logits,
                                    batch["question_classes"],
                                    answer_vocabs=self.answer_vocabs,
                                    q_types_mapping=self.q_types_mapping,
                                    q_types=self.q_types,
                                    task_heads=self.task_heads)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

            # Optional validation
            if val_loader and (epoch + 1) % validate_every == 0:
                val_loss = self.evaluate(val_loader=val_loader,disease_model=disease_model)
                #val_loss = self.eval(val_loader=val_loader,disease_model=disease_model)
                print(f"Validation Loss: {val_loss:.4f}")

                # Save best model
                if save_best_path and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.state_dict(), save_best_path)
                    print(f"✅ Best model saved to {save_best_path}")

        print("=== Fine-tuning Completed ===")
    

    @torch.no_grad()
    def evaluate(self, val_loader, disease_model=None):
        """Evaluate model on the validation loader and return average loss."""
        #self.eval()  # sets model to evaluation mode
        total_loss = 0.0
        total_batches = 0

        print("🔍 Evaluating on validation set...")
        for batch in val_loader:
            preds, answers, task_logits = forward_batch(
                batch["images"],
                batch["input_ids"],
                batch["attention_mask"],
                batch["question"],
                batch["answers"],
                batch["question_classes"],
                qtype_classifier=self.qtype_classifier,
                fusion_module=self.fusion_module,
                q_types=self.q_types,
                q_types_mapping=self.q_types_mapping,
                task_heads=self.task_heads,
                device=self.device,
                image_encoder=self.image_encoder,
                question_encoder=self.question_encoder,
                tokenizer=self.tokenizer,
                disease_model=disease_model
            )

            loss = compute_loss(
                preds,
                answers,
                task_logits,
                batch["question_classes"],
                answer_vocabs=self.answer_vocabs,
                q_types_mapping=self.q_types_mapping,
                q_types=self.q_types,
                task_heads=self.task_heads
            )

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        print(f"✅ Validation Loss: {avg_loss:.4f}")
        self.train()  # switch back to training mode
        return avg_loss


    def fine_tune_model(self, epochs, data_train, train_loader, disease_model, checkpoint_path):
        """
        Fine-tunes an already trained model using previous weights.

        Args:
            epochs (int): Number of fine-tuning epochs
            data_train (Dataset): Training dataset
            train_loader (DataLoader): Training data loader
            disease_model (nn.Module): Disease classifier module
            checkpoint_path (str): Path to pretrained model checkpoint (.pt or .pth)
        """

        print(f"🔁 Loading pretrained weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Rebuild all required components
        self.epochs = epochs
        self.train_data = data_train
        self.train_loader = train_loader
        self.answer_vocabs = build_answer_vocab(self.train_data, self.q_types_mapping)
        self.task_vocabs = build_vocabs(self.train_data, self.q_types_mapping)

        # Model architecture (same as in train_model)
        self.qtype_classifier = QuestionTypeClassifier(num_types=len(self.q_types)).to(self.device)
        self.answer_classifier = nn.Linear(self.hidden_dim, len(self.answer_vocabs)).to(self.device)
        self.fusion_module = CoAttentionFusion(
            img_dim=self.img_dim,
            ques_dim=self.ques_dim,
            disease_dim=self.disease_dim,
            hidden_dim=self.hidden_dim,
            answer_vocab=self.answer_vocabs
        ).to(self.device)

        # ✅ Load pretrained weights safely (ignore mismatches for vocab size)
        def safe_load(target, source_dict, name):
            try:
                target.load_state_dict(source_dict[name])
                print(f"✓ Loaded weights for {name}")
            except Exception as e:
                print(f"[Warning] Skipping partial load for {name}: {e}")

        if "image_encoder" in checkpoint:
            safe_load(self.image_encoder, checkpoint, "image_encoder")
        if "question_encoder" in checkpoint:
            safe_load(self.question_encoder, checkpoint, "question_encoder")
        if "fusion_module" in checkpoint:
            safe_load(self.fusion_module, checkpoint, "fusion_module")
        if "qtype_classifier" in checkpoint:
            safe_load(self.qtype_classifier, checkpoint, "qtype_classifier")
        if "task_heads" in checkpoint:
            safe_load(self.task_heads, checkpoint, "task_heads")

        print("✅ All available weights loaded. Starting fine-tuning...")

        # ✅ Optimizer — includes all modules (full fine-tuning, no freezing)
        self.optimizer = torch.optim.AdamW(
            list(self.fusion_module.parameters()) +
            list(self.question_encoder.parameters()) +
            list(self.image_encoder.parameters()) +
            list(self.qtype_classifier.parameters()),
            lr=1e-5  # Slightly lower LR for fine-tuning
        )

        # ✅ Training loop (same as base train_model)
        for epoch in range(self.epochs):
            self.fusion_module.train()
            self.qtype_classifier.train()
            total_loss = 0

            print(f"🧠 Epoch {epoch + 1}/{self.epochs}")
            for batch in self.train_loader:
                self.optimizer.zero_grad()

                preds, answers, task_logits = forward_batch(
                    batch["images"],
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["question"],
                    batch["answers"],
                    batch["question_classes"],
                    qtype_classifier=self.qtype_classifier,
                    fusion_module=self.fusion_module,
                    q_types=self.q_types,
                    q_types_mapping=self.q_types_mapping,
                    task_heads=self.task_heads,
                    device=self.device,
                    image_encoder=self.image_encoder,
                    question_encoder=self.question_encoder,
                    tokenizer=self.tokenizer,
                    disease_model=disease_model
                )

                loss = compute_loss(
                    preds,
                    answers,
                    task_logits,
                    batch["question_classes"],
                    answer_vocabs=self.answer_vocabs,
                    q_types_mapping=self.q_types_mapping,
                    q_types=self.q_types,
                    task_heads=self.task_heads
                )

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs} | Fine-tune Loss: {total_loss / len(train_loader):.4f}")
        
        print("✅ Fine-tuning complete.")


    def eval(self, val_loader,disease_model):
        """
        Evaluate the model on the validation set.
    
        Args:
            val_loader: DataLoader for validation data.
    
        Returns:
            avg_loss: average validation loss
            all_preds: list of predicted labels
            all_answers: list of ground truth answers
        """
        self.fusion_module.eval()
        self.question_encoder.eval()
        self.image_encoder.eval()
        self.qtype_classifier.eval()
        for head in self.task_heads.values():
            head.eval()
    
        total_loss = 0.0
        all_preds, all_answers = [], []
    
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                answers = batch["answers"]
                q_classes = batch["question_classes"]
    
                # ---- Disease vector ----
                disease_vec = disease_model(images)
    
                # ---- Question type classifier ----
                task_logits = self.qtype_classifier(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )  # [B, num_types]
    
                # map fine-grained → general
                mapped_classes = [
                    self.q_types_mapping[c[0] if isinstance(c, list) else c]
                    for c in q_classes
                ]
    
                # ---- Encoders ----
                q_feat = self.question_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).pooler_output  # [B, 768]
    
                img_outputs = self.image_encoder(pixel_values=images)
                img_feat = img_outputs.last_hidden_state  # [B, R, 768]
    
                # ---- Fusion ----
                fused = self.fusion_module(img_feat, q_feat, disease_vec)
    
                # ---- Predict per sample ----
                pred_tensors = []
                batch_preds = []
                for i, task_type in enumerate(mapped_classes):
                    predictor = self.task_heads[task_type]
                    #pred_out = predictor(fused[i].unsqueeze(0))
                    pred_tensor = predictor(fused[i].unsqueeze(0))   # shape [1, C] or [1,1] for count
                    pred_tensors.append(pred_tensor)
    
                    if task_type == "yesno":
                        pred_label = "Yes" if torch.argmax(pred_tensor, dim=1).item() == 1 else "No"
                    elif task_type == "count":
                        pred_val = pred_tensor.squeeze()
                        pred_label = str(int(round(pred_val.item())))
                        #pred_label = str(int(pred_out.item()))
                    else:
                        ans_idx = torch.argmax(pred_tensor, dim=1).item()
                        if task_type in self.answer_vocabs and ans_idx < len(self.answer_vocabs[task_type]):
                            inv_vocab = {v: k for k, v in self.answer_vocabs[task_type].items()}
                            pred_label = inv_vocab.get(ans_idx, str(ans_idx))
                        else:
                            pred_label = str(ans_idx)
    
                    batch_preds.append(pred_label)
    
                # ---- Compute loss ----
                """
                batch_loss = compute_loss(
                    [self.task_heads[c](fused[i].unsqueeze(0)) for i, c in enumerate(mapped_classes)],
                    answers,
                    task_logits,
                    q_classes,
                    self.answer_vocabs
                )"""
                # compute batch loss using the same preds (tensors) and required extra args
                batch_loss = compute_loss(
                    preds=pred_tensors,
                    answers=answers,
                    task_logits=task_logits,
                    true_q_classes=q_classes,
                    answer_vocabs=self.answer_vocabs,
                    q_types_mapping=self.q_types_mapping,
                    q_types=self.q_types,
                    task_heads=self.task_heads
                )
                total_loss += batch_loss.item()
    
                all_preds.extend(batch_preds)
                all_answers.extend(answers)
    
        avg_loss = total_loss / len(val_loader)
        return avg_loss, all_preds, all_answers

    
    def load1(self,load_path = "vqa_model.pt"):
        checkpoint = torch.load(load_path, map_location=self.device,weights_only=False)
        self.task_vocabs=checkpoint["task_vocabs"]
        self.answer_vocabs=checkpoint["answer_vocabs"]
        self.fusion_module = CoAttentionFusion(
            img_dim=self.img_dim, ques_dim=self.ques_dim, disease_dim=self.disease_dim, hidden_dim=self.hidden_dim,
            answer_vocab=checkpoint["answer_vocabs"]
        ).to(self.device)
        self.fusion_module.load_state_dict(checkpoint["fusion_module"])    
        self.question_encoder.load_state_dict(checkpoint["question_encoder"])
        self.image_encoder.load_state_dict(checkpoint["image_encoder"])
        self.qtype_classifier.load_state_dict(checkpoint["qtype_classifier"])
        
        for k, v in checkpoint["task_heads"].items():
            self.task_heads[k].load_state_dict(v)
            
        # 3. Recreate optimizer with correct params
        self.optimizer = torch.optim.AdamW(
            list(self.fusion_module.parameters()) + 
            list(self.question_encoder.parameters()) + 
            list(self.image_encoder.parameters()) + 
            list(self.qtype_classifier.parameters()), 
            lr=2e-5
        )
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print("Model and components loaded successfully")
    
    def save(self,save_path = "vqa_model.pt"):
        torch.save({
            "fusion_module": self.fusion_module.state_dict(),
            "question_encoder": self.question_encoder.state_dict(),
            "image_encoder": self.image_encoder.state_dict(),
            "qtype_classifier": self.qtype_classifier.state_dict(),
            "task_heads": {k: v.state_dict() for k, v in self.task_heads.items()},
            "optimizer": self.optimizer.state_dict(),
            "epochs": self.epochs,
            "answer_vocabs": self.answer_vocabs,
            "task_vocabs": self.task_vocabs
        }, save_path)
        print(f"Model saved at {save_path}")

    def predict(self,disease_model, image, question):
        self.fusion_module.eval()
        self.question_encoder.eval()
        self.image_encoder.eval()
        self.qtype_classifier.eval()
    
        with torch.no_grad():
            # ---- Preprocess image ----
            image_tensor = preprocess_image(image).unsqueeze(0).to(self.device)
    
            # ---- Disease vector ----
            if(disease_model):
                disease_vec = diseasem(disease_model,image_tensor)
            else:
                disease_vec = disease_model(image_tensor)
    
            # ---- Encode question ----
            q_inputs = router_tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(self.device)
    
            # DistilBERT classifier for q-type
            task_logits = self.qtype_classifier(
                input_ids=q_inputs["input_ids"],
                attention_mask=q_inputs["attention_mask"]
            )  # [1, num_types]
    
            task_idx = torch.argmax(task_logits, dim=1).item()
            task_type = self.q_types[task_idx]  # map index → general type
    
            # ---- Question encoder for fusion ----
            q_feat = self.question_encoder(**q_inputs).pooler_output  # [1, 768]
    
            # ---- Image encoder ----
            img_outputs = self.image_encoder(pixel_values=image_tensor)
            img_feat = img_outputs.last_hidden_state  # [1, R, 768]
    
            # ---- Fusion ----
            fused = self.fusion_module(img_feat, q_feat, disease_vec)
    
            # ---- Task-specific head ----
            predictor = self.task_heads[task_type]  # use pretrained head
            pred_out = predictor(fused)
    
            # ---- Decode prediction ----
            if task_type == "yesno":
                pred_label = "Yes" if torch.argmax(pred_out, dim=1).item() == 1 else "No"
    
            elif task_type == "count":
                pred_label = str(int(pred_out.item()))
    
            else:  # categorical answer
                ans_idx = torch.argmax(pred_out, dim=1).item()
                if task_type in self.answer_vocabs and ans_idx < len(self.answer_vocabs[task_type]):
                    inv_vocab = {v: k for k, v in self.answer_vocabs[task_type].items()}
                    pred_label = inv_vocab.get(ans_idx, str(ans_idx))
                else:
                    pred_label = str(ans_idx)
    
        return pred_label

    def load(self, load_path="vqa_model.pt", strict=True, load_optimizer=True):
        print(f"🔁 Loading pretrained weights from: {load_path}")
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)

        # ---- Load stored vocabs ----
        self.task_vocabs = checkpoint["task_vocabs"]
        self.answer_vocabs = checkpoint["answer_vocabs"]

        # ---- Rebuild fusion module with vocab ----
        self.fusion_module = CoAttentionFusion(
            img_dim=self.img_dim,
            ques_dim=self.ques_dim,
            disease_dim=self.disease_dim,
            hidden_dim=self.hidden_dim,
            answer_vocab=self.answer_vocabs
        ).to(self.device)

        # ---- Load model weights ----
        self.fusion_module.load_state_dict(checkpoint["fusion_module"], strict=strict)
        self.question_encoder.load_state_dict(checkpoint["question_encoder"], strict=strict)
        self.image_encoder.load_state_dict(checkpoint["image_encoder"], strict=strict)
        self.qtype_classifier.load_state_dict(checkpoint["qtype_classifier"], strict=strict)

        for k, v in checkpoint["task_heads"].items():
            if k in self.task_heads:
                self.task_heads[k].load_state_dict(v, strict=False)

        # ---- Recreate optimizer ----
        self.optimizer = torch.optim.AdamW(
            list(self.fusion_module.parameters()) +
            list(self.question_encoder.parameters()) +
            list(self.image_encoder.parameters()) +
            list(self.qtype_classifier.parameters()),
            lr=2e-5
        )

        # ---- Try to load optimizer if compatible ----
        if load_optimizer and "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                print("✅ Optimizer state loaded successfully.")
            except ValueError as e:
                print("⚠️ Optimizer state incompatible, using freshly initialized optimizer.")
                print("Reason:", e)

        print("✅ Model and components loaded successfully.")
