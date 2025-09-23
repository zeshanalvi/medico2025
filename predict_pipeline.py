def vqa_pipeline(image, question):
    # 1. Disease classification
    disease_vec = disease_model(image)
    #disease_vec = disease_model(image.unsqueeze(0).to(device))  # (1,23)

    # 2. Question type classification
    q_type_idx = classify_question_type(question)
    q_types = ["yesno", "single", "multi", "color", "location", "count"]
    task_type = q_types[q_type_idx]

    # 3. Image + Question embeddings
    img_emb = image_encoder(pixel_values=image.unsqueeze(0).to(device)).pooler_output
    q_inputs = router_tokenizer(question, return_tensors="pt", truncation=True).to(device)
    q_emb = question_encoder(**q_inputs).pooler_output

    # 4. Fusion
    img_emb = img_emb.float()
    q_emb = q_emb.float()
    disease_vec = disease_vec.float()
    dis_vec = disease_vec.unsqueeze(0).float().to(device)
    #print("Image\t",img_emb.shape,img_emb.device)
    #print("Question\t",q_emb.shape,q_emb.device)
    #print("Disease\t",dis_vec.shape,dis_vec.device)
    
    fusion_module = CoAttentionFusion(img_dim=768, ques_dim=768, disease_dim=23, hidden_dim=512).to(device)
    fused=fusion_module.forward(img_emb, q_emb, dis_vec)
    #fused = FusionModule()(img_emb, q_emb, disease_vec)

    # 5. Task-specific predictor
    predictor = TaskPredictor(task_type).to(device)
    pred_out = predictor(fused)

    #print("Prediction\t",pred_out.shape,pred_out)
    print("Task Type\t",task_type)

    # Convert prediction to text for generator
    if task_type == "yesno":
        pred_label = "Yes" if torch.argmax(pred_out) == 1 else "No"
    elif task_type == "count":
        pred_label = f"{pred_out.item():.0f}"
    else:
        pred_label = str(torch.argmax(pred_out).item())

    print("Label\t",pred_label)
    # 6. Generate descriptive answer
    answer = generate_descriptive_answer(question, pred_label, fused)
    return answer