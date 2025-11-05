from transformers import ViTModel, BertModel
import torch.nn.functional as F

import torch
import torch.nn as nn

#image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
#question_encoder = BertModel.from_pretrained("bert-base-uncased").to(device)


class CoAttentionFusion1(nn.Module):
    def __init__(self, img_dim, ques_dim, disease_dim, hidden_dim, answer_vocab):
        super(CoAttentionFusion, self).__init__()
        
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.ques_proj = nn.Linear(ques_dim, hidden_dim)
        self.dis_proj = nn.Linear(disease_dim, hidden_dim)

        self.att_img = nn.Linear(hidden_dim, 1)
        self.att_dis = nn.Linear(hidden_dim, 1)
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)

        # ✅ Store answer vocab inside model for later use
        self.answer_vocab = answer_vocab

    def forward(self, img_feat, ques_feat, dis_vec):
        # Project features
        #print("Input Shapes\t",img_feat.shape, ques_feat.shape, dis_vec.shape)
        img_proj = torch.tanh(self.img_proj(img_feat))     # [B, H]
        ques_proj = torch.tanh(self.ques_proj(ques_feat))  # [B, H]

        #print("Ques_proj",ques_proj.shape,img_proj.shape)
        
        dis_vec = dis_vec.to(torch.float32)
        dis_proj = torch.tanh(self.dis_proj(dis_vec))      # [B, H]


        #print("After projection\t",img_proj.shape, ques_proj.shape, dis_proj.shape)

        # Expand question for image alignment
        #ques_expand = ques_proj.unsqueeze(1).expand_as(img_proj)
        
        
        #ques_expand = ques_proj#.expand_as(img_proj)
        #img_co = img_proj * ques_expand

        #Replacement of above 2 lines

        ques_proj = ques_proj.unsqueeze(1)                    # [16, 1, 512]
        ques_expand = ques_proj.expand(-1, img_proj.size(1), -1)  # [16, 197, 512]
        img_co = img_proj * ques_expand                       # [16, 197, 512]

        #print("ques_expand",ques_expand.shape,img_co.shape)
        
        att_img_weights = torch.sigmoid(self.att_img(img_co))  # [B, 1]
        #img_att = att_img_weights * img_proj
        img_att = (att_img_weights * img_proj).sum(1)
        #att_img_weights = F.softmax(self.att_img(img_co), dim=1)   # [B, R, 1]
        #img_att = (att_img_weights * img_proj).sum(1)              # [B, H]

        # Co-attention with disease vector
        dis_co = dis_proj * ques_proj
        att_dis_weights = torch.sigmoid(self.att_dis(dis_co))      # [B, 1]
        dis_att = att_dis_weights * dis_proj                       # [B, H]


        #print(img_att.shape, ques_proj.shape, dis_att.shape)


        if img_att.dim() == 1:
            img_att = img_att.unsqueeze(0)   # [1, H]
        if ques_proj.dim() == 1:
            ques_proj = ques_proj.unsqueeze(0)     # [1, H]
        if dis_att.dim() == 1:
            dis_att = dis_att.unsqueeze(0)     # [1, H]


        # Concatenate

        ques_proj_flat = ques_proj.squeeze(1)
        dis_att_flat = dis_att.squeeze(1) 

        #print(img_att.shape,ques_proj.shape,dis_att.shape)
        
        joint_feat = torch.cat([img_att, ques_proj_flat, dis_att_flat], dim=1)  # [B, 3H]
        fused = torch.tanh(self.fusion(joint_feat))  # [B, H]
        return fused
    
class CoAttentionFusion(nn.Module):
    def __init__(self, img_dim, ques_dim, disease_dim, hidden_dim, answer_vocab):
        super(CoAttentionFusion, self).__init__()
        
        # Dimensions based on your data: 
        # img_dim=768, ques_dim=768, disease_dim=23, answer_vocab=6

        #print("Hidden Dimensions",hidden_dim)
        
        # 1. Projection Layers
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.ques_proj = nn.Linear(ques_dim, hidden_dim)
        self.dis_proj = nn.Linear(disease_dim, hidden_dim)

        # 2. Attention Scoring Layers (Output dim = 1)
        self.att_img = nn.Linear(hidden_dim, 1) # Scores for each of the 197 image tokens
        self.att_dis = nn.Linear(hidden_dim, 1) # Score for the single disease vector

        # 3. Fusion Layer (Combines 3 * hidden_dim into 1 * hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)

        self.answer_vocab = answer_vocab
        # NOTE: You'll likely need a final prediction layer after fusion, e.g.,
        # self.classifier = nn.Linear(hidden_dim, answer_vocab)

    def forward(self, img_feat, ques_feat, dis_vec):
        #print("Input Shapes\t",img_feat.shape, ques_feat.shape, dis_vec.shape)

        B = img_feat.size(0) # Batch size 8
        R = img_feat.size(1) # Image regions/tokens 197
        Q = ques_feat.size(1) # Question tokens 28
        
        # 1. Project Features to Common Hidden Space (Hidden_dim: H)
        # img_feat: [B, R, 768] -> img_proj: [B, R, H]
        img_proj = torch.tanh(self.img_proj(img_feat))
        
        # ques_feat: [B, 768] -> ques_proj: [B, 28, H]
        ques_proj = torch.tanh(self.ques_proj(ques_feat))
        
        # dis_vec: [B, 23] -> dis_proj: [B, H]
        dis_vec = dis_vec.to(torch.float32) 
        dis_proj = torch.tanh(self.dis_proj(dis_vec))
        
        # --- Image Co-Attention (Question attends to Image tokens) ---

        #print("Dimensions\t",img_proj.shape,ques_proj.shape,dis_proj.shape)


        ques_mean=ques_proj.mean(dim=1,keepdim=True) # [B,1,H]
        #print("[B,1,H]",ques_mean.shape)
        img_co=img_proj*ques_mean
        #print("[B,197,H]",img_co.shape)

        att_img_weights = F.softmax(self.att_img(img_co), dim=1)  # [B, 197, 1]
        #print("[B, 197, 1]",att_img_weights.shape)
        img_att = (att_img_weights * img_proj).sum(dim=1)         # [B, H]
        #print("[B, H]",img_att.shape)

        # ---- Co-attention between Disease and Question ----
        dis_co = dis_proj.unsqueeze(1) * ques_proj                # [B, 28, H]
        #print("[B, 28, H]",dis_co.shape)
        att_dis_weights = F.softmax(self.att_dis(dis_co), dim=1)  # [B, 28, 1]
        #print("[B, 28, 1]",att_dis_weights.shape)
        dis_att = (att_dis_weights * ques_proj).sum(dim=1)        # [B, H]
        #print("[B, H]",dis_att.shape)

        # ---- Fusion ----
        joint_feat = torch.cat([img_att, dis_att, dis_proj], dim=1)  # [B, 3H]
        #print("[B, 3H]",joint_feat.shape)
        fused = torch.tanh(self.fusion(joint_feat))                  # [B, H]
        #print("[B, H]",fused.shape)

        
        
        # Expand question features to match image token dimension
        # ques_proj: [B, H] -> [B, 1, H] -> [B, R, H]
        #ques_expand = ques_proj.unsqueeze(1).expand(-1, R, -1)
        
        # Element-wise product for co-attention features (gating)
        # img_co: [B, R, H]
        #img_co = img_proj * ques_expand
        
        # Calculate attention scores for each image token
        # raw_att_weights: [B, R, 1]
        #raw_att_weights = self.att_img(img_co)
        
        # Normalize weights across the 197 tokens (dim=1)
        # att_img_weights: [B, R, 1]
        #att_img_weights = F.softmax(raw_att_weights, dim=1)
        
        # Apply weights and sum across tokens to get a single image vector
        # img_att: [B, H]
        #img_att = (att_img_weights * img_proj).sum(1)
        
        # --- Disease Co-Attention (Question attends to Disease vector) ---
        
        # Element-wise product for co-attention features (gating)
        # Since ques_proj and dis_proj are both [B, H], we can multiply directly
        # dis_co: [B, H]
        #dis_co = ques_proj * dis_proj 
        
        # Calculate attention score (sigmoid is okay here since there's only 1 feature)
        # att_dis_weights: [B, 1]
        #att_dis_weights = torch.sigmoid(self.att_dis(dis_co))
        
        # Apply score to the disease feature (we don't sum since it's already one vector)
        # dis_att: [B, H]
        #dis_att = att_dis_weights * dis_proj
        
        # --- Final Fusion ---
        
        # Concatenate the three final attended/projected features
        # joint_feat: [B, 3H]
        #joint_feat = torch.cat([img_att, ques_proj, dis_att], dim=1)
        
        # Pass through the fusion layer
        # fused: [B, H]
        #fused = torch.tanh(self.fusion(joint_feat))
        
        return fused