import torch 
from torch.nn import CrossEntropyLoss, MSELoss
import re
from .models import disease_model, diseasem
from transformers import BertModel
import torch

#!pip install rouge_score

#from rouge_score import rouge_scorer

from nltk.translate.meteor_score import meteor_score


def forward_batch(images, input_ids, attention_mask, question, answers, question_classes=None,
                  qtype_classifier=None,fusion_module=None,q_types=None,q_types_mapping=None,
                  task_heads=None,device=None,image_encoder=None,question_encoder=None,tokenizer=None,
                  disease_model=disease_model):
    # Image encoding
    img_outputs = image_encoder(pixel_values=images.to(device))
    img_feat = img_outputs.last_hidden_state  # [B, R, 768]
    flattened_img_feat = img_feat.view(img_feat.size(0), -1) #[B, something]

    # Question encoding (DistilBERT for qtype classification)
    task_logits = qtype_classifier(input_ids=input_ids.to(device),attention_mask=attention_mask.to(device))  # [B, num_types]

    #print(type(input_ids),type(question))
    #print("Questions\n",question,"\n\n\n")
    bert_question = tokenizer(question,padding=True, truncation=True,return_tensors="pt")
    #print("Tockenized Questions",bert_question)
    
    # 2. Move the input tensors to the same device
    input_ids = bert_question['input_ids'].to(device)
    attention_mask = bert_question['attention_mask'].to(device)
    
    # The model call
    with torch.no_grad(): # Disable gradient calculations since we're not training
        # The ** operator unpacks the dictionary into keyword arguments
        bert_question_last = question_encoder(input_ids=input_ids,attention_mask=attention_mask)
    # The outputs are typically a tuple, with the primary output being the last hidden state
    encoded_questions = bert_question_last.last_hidden_state
    #print(f"✅ Successful Model Call!")
    #print(f"Batch size: {encoded_questions.shape[0]}")
    #print(f"Sequence length: {encoded_questions.shape[1]}")
    #print(f"Hidden size: {encoded_questions.shape[2]} (768 for bert-base-uncased)")
    #print(encoded_questions.shape)


    # Use another encoder for question embeddings (router encoder you already had)
    #q_feat = question_encoder(input_ids=input_ids.to(device),attention_mask=attention_mask.to(device)).pooler_output  # [B, 768]
    #q_feat = question_encoder(input_ids=encoded_questions.to(device),attention_mask=attention_mask.to(device)).pooler_output  # [B, 768]

    # Disease model
    
    if(disease_model):
        disease_vec = diseasem(disease_model,images.to(device))  # [B, 23]
    else:
        disease_vec = disease_model(images.to(device))  # [B, 23]

    
    #print("task_logits-num_types",type(task_logits))
    #print("img_feat",type(img_feat))
    #print("q_feat-768",type(encoded_questions))
    #print("disease_vec",type(disease_vec))
    #print("task_logits-num_types",task_logits.shape)
    #print("img_feat",img_feat.shape)
    #print("q_feat-768",encoded_questions.shape)
    #print("disease_vec",disease_vec.shape)

    # Fusion
    fused = fusion_module(img_feat, encoded_questions, disease_vec)
    #fused = fusion_module(flattened_img_feat, q_feat, disease_vec)

    #print("Fused Dimensions",fused.shape)    

    # Task-specific predictions (list of preds per sample, like before)
    preds = []
    for i, q_class in enumerate(question_classes):#q_class from task_type
        #print("working on\t",i,"\t",q_class)
        mapped_type = q_types_mapping[q_class[0] if isinstance(q_class, list) else q_class]
        predictor = task_heads[mapped_type]   # ✅ trained head
        pred_out = predictor(fused[i].unsqueeze(0),question)
        preds.append(pred_out)
        #general_class = q_types_mapping[task_type[0] if isinstance(task_type, list) else task_type]
        #head = TaskPredictor(general_class, hidden=fused.size(-1)).to(device)
        #preds.append(head(fused[i].unsqueeze(0)))

    return preds, answers, task_logits

def extract_count(answer_str):
    """
    Try to convert an answer string into a number.
    Returns None if it cannot be parsed.
    """
    try:
        # Direct numeric
        return float(answer_str)
    except ValueError:
        pass

    # Handle words like "one", "two", etc.
    word2num = {
        "zero": 0, "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    tokens = answer_str.lower().split()
    for t in tokens:
        if t in word2num:
            return float(word2num[t])

    # Extract any digits from the string
    numbers = re.findall(r"\d+", answer_str)
    if numbers:
        return float(numbers[0])

    return None  # fallback


def compute_meteor(preds, answers, answer_vocabs, mapped_classes):
    scores = []
    for pred, ans, c in zip(preds, answers, mapped_classes):
        if c not in answer_vocabs:
            continue
        # Get predicted index
        pred_idx = pred.argmax(dim=1).item()
        # Map index back to string
        inv_vocab = {v: k for k, v in answer_vocabs[c].items()}
        pred_str = inv_vocab.get(pred_idx, "")
        # METEOR score between predicted and ground truth answer
        score = meteor_score([ans.split()], pred_str.split())
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def compute_meteor_bool(preds, answers, answer_vocabs, mapped_classes):
    scores = []
    for pred, ans, c in zip(preds, answers, mapped_classes):
        if c not in answer_vocabs:
            continue

        # Handle sequence logits [seq_len, vocab_size] or [1, seq_len, vocab_size]
        if pred.dim() == 3:
            pred = pred.squeeze(0)  # remove batch dim if present → [seq_len, vocab_size]
        if pred.dim() == 2:
            pred = pred[-1]  # take logits of the last token → [vocab_size]

        # Get predicted index
        pred_idx = pred.argmax(dim=-1).item()

        # Map index back to string
        inv_vocab = {v: k for k, v in answer_vocabs[c].items()}
        pred_str = inv_vocab.get(pred_idx, "")

        # Compute METEOR between predicted and actual answer
        score = meteor_score([ans.split()], pred_str.split())
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0



def compute_rouge(preds, answers, answer_vocabs, mapped_classes):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ans, c in zip(preds, answers, mapped_classes):
        if c not in answer_vocabs:
            continue
        pred_idx = pred.argmax(dim=1).item()
        inv_vocab = {v: k for k, v in answer_vocabs[c].items()}
        pred_str = inv_vocab.get(pred_idx, "")
        score = scorer.score(ans, pred_str)["rougeL"].fmeasure
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0

def compute_loss(preds, answers, task_logits, true_q_classes, answer_vocabs,q_types_mapping,q_types,task_heads):
    """
    preds: list of model predictions for each sample
    answers: list of strings (descriptive answers)
    task_logits: tensor [batch_size, num_task_types]
    true_q_classes: list of lists (fine-grained classes for each question)
    answer_vocabs: dict mapping {q_type: {answer: index}}
    """

    ce_loss = CrossEntropyLoss()
    mse_loss = MSELoss()

    total_loss = 0

    # 1) Map fine-grained → general classes
    mapped_classes = [
        q_types_mapping[c[0] if isinstance(c, list) else c]
        for c in true_q_classes
    ]

    # 2) Question type classification loss
    true_task_types = torch.tensor(
        [q_types.index(c) for c in mapped_classes],
        device=task_logits.device
    )

    #print("task_logits, true_task_types\t",task_logits, true_task_types)
    
    #print("task_logits, true_task_types\t",task_logits.shape, true_task_types.shape)
    
    task_loss = ce_loss(task_logits, true_task_types)
    total_loss += task_loss

    # 3) Answer prediction loss (per sample)
    for pred, ans, c in zip(preds, answers, mapped_classes):
        #print("Answers\n","type:",c,"\nPredicted Answer:\t",pred,"\nActual Answer:\t",ans,"\n")
        predictor = task_heads[c]   # ✅ trained head
        if c == "count":
            # For count, answer must be numeric
            try:
                ans_val = float(ans)
                ans_val = torch.tensor([ans_val], device=pred.device)
                total_loss += mse_loss(pred.squeeze(), ans_val)
            except ValueError:
                print(f"[Warning] Skipping non-numeric count answer: {ans}")
                continue

        else:
            # For categorical tasks (yesno, single, multi, etc.)
            if ans not in answer_vocabs.get(c, {}):
                print(f"[Warning] Skipping unseen or descriptive answer {ans} for task {c}")
                continue

            ans_idx = answer_vocabs[c][ans]

            #print("Before error pred",type(pred),pred)
            #print("ans_idx",ans_idx)

            if ans_idx >= pred.size(1):
                print(f"[Warning] Skipping answer {ans} for task {c}: "
                      f"index {ans_idx} >= pred.size(1)")
                continue

            ans_tensor = torch.tensor([ans_idx], device=pred.device)
            pred = pred[:, -1, :]

            # Adding loss here becasue for the next type will have some other loss.
            total_loss += ce_loss(pred, ans_tensor)

    meteor = compute_meteor_bool(preds, answers, answer_vocabs, mapped_classes)
    print(f"Validation METEOR: {meteor:.4f}")
    #rouge = compute_rouge(preds, answers, answer_vocabs, mapped_classes)
    #print(f"Validation ROUGE-L: {rouge:.4f}")

    
    return total_loss / len(preds)