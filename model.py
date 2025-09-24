import torch
import torch.nn as nn
import os
from .qtype import QuestionTypeClassifier
from .functions import build_vocabs, build_answer_vocab, collate_fn, preprocess_example, normalize_answer, preprocess_image
from .models import disease_model, device, generate_descriptive_answer, router_tokenizer, gen_model
from .tpred import TaskPredictor
from .model_functions import compute_loss, compute_meteor, compute_rouge, extract_count, forward_batch
from .fussionmodel import BertModel, CoAttentionFusion, ViTModel, F


class VQAModel(nn.Module):
    def __init__(self,img_dim, ques_dim, disease_dim, hidden_dim):
        super(VQAModel, self).__init__()
        #self.fusion = CoAttentionFusion(img_dim, ques_dim, disease_dim, hidden_dim, answer_vocab=answer_vocab)
        self.qtype_classifier=None
        self.answer_classifier=None
        self.epochs=1
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim=hidden_dim
        self.input_dim=768
        self.ques_dim=ques_dim
        self.disease_dim=disease_dim
        self.img_dim=img_dim
        self.fusion_module=None
        self.question_encoder=BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.image_encoder=ViTModel.from_pretrained("google/vit-base-patch16-224").to(self.device)
        self.optimizer=None
        self.answer_vocabs=None
        self.task_vocabs=None
        self.data_train=None
        self.train_loader=None
        self.q_types = ["yesno", "single", "multi", "color", "location", "count"]
        # Create task-specific heads (trainable)
        self.task_heads = nn.ModuleDict({
            t: TaskPredictor(t, hidden=hidden_dim) for t in self.q_types
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
        
    
    def train(self,epochs,data_train,train_loader):
        self.epochs=epochs
        self.train_data=train_data
        self.train_loader=train_loader
        self.answer_vocabs = build_answer_vocab(self.train_data, self.q_types_mapping)
        self.task_vocabs=build_vocabs(self.train_data,self.q_types_mapping)
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
        for epoch in range(self.epochs):
            self.fusion_module.train()
            self.qtype_classifier.train()
            total_loss = 0
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                preds, answers, task_logits = forward_batch(
                    batch["images"],
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["answers"],
                    batch["question_classes"],  # fine-grained from dataset
                    qtype_classifier=self.qtype_classifier,
                    fusion_module=self.fusion_module,
                    q_types=self.q_types,
                    q_types_mapping=self.q_types_mapping,
                    task_heads=self.task_heads,
                    device=self.device
                )
                #preds, answers = forward_batch(batch["images"],batch["input_ids"], batch["attention_mask"], batch["answers"], batch["question_classes"])
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
    
    def eval(self, val_loader):
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

    def eval_old(self,val_loader):
        fusion_module.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in val_loader:
                preds, answers, task_logits = forward_batch(
                    batch["images"],
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["answers"],
                    batch["question_classes"],
                    device=self.device
                )
                loss = compute_loss(
                    preds,
                    answers,
                    task_logits,
                    batch["question_classes"],
                    self.answer_vocabs
                )
                total_loss += loss.item()
            print(f"Validation Loss: {total_loss / len(val_loader)}")

    def load(self,load_path = "vqa_model.pt"):
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

    def predict(self, image, question):
        self.fusion_module.eval()
        self.question_encoder.eval()
        self.image_encoder.eval()
        self.qtype_classifier.eval()
    
        with torch.no_grad():
            # ---- Preprocess image ----
            image_tensor = preprocess_image(image).unsqueeze(0).to(self.device)
    
            # ---- Disease vector ----
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

