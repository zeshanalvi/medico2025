from transformers import ViTModel, BertModel
import torch.nn.functional as F

image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
question_encoder = BertModel.from_pretrained("bert-base-uncased").to(device)

class CoAttentionFusion(nn.Module):
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

        print("Ques_proj",ques_proj.shape,img_proj.shape)
        
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