import torch
import torch.nn as nn
from losses.loss import MSELoss, CELoss

class Attention(nn.Module):
    def __init__(self, audio_dim, text_dim, video_dim, output_dim1, output_dim2=1, layers='256,128', dropout=0.3):
        super(Attention, self).__init__()

        self.audio_mlp = self.MLP(audio_dim, layers, dropout)
        self.text_mlp  = self.MLP(text_dim,  layers, dropout)
        self.video_mlp = self.MLP(video_dim, layers, dropout)

        layers_list = list(map(lambda x: int(x), layers.split(',')))
        hiddendim = layers_list[-1] * 2
        self.attention_mlp = self.MLP(hiddendim, layers, dropout)

        self.fc_att   = nn.Linear(layers_list[-1], 2)
        self.fc_out_v = nn.Linear(layers_list[-1], output_dim1)
        self.fc_out_a = nn.Linear(layers_list[-1], output_dim1)
        self.fc_out_b = nn.Linear(layers_list[-1], output_dim1)
        self.fc_out_2 = nn.Linear(layers_list[-1], output_dim2)
        self.emo_loss = CELoss()
    
    def MLP(self, input_dim, layers, dropout):
        all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        module = nn.Sequential(*all_layers)
        return module

    def forward(self, audio_feat, text_feat, video_feat, emotions, train=False):
        video_hidden = self.video_mlp(text_feat) # [32,1024] -> [32,128]
        audio_hidden = self.audio_mlp(audio_feat) # [32,1024] -> [32,128]
  
        multi_hidden1 = torch.cat([audio_hidden, video_hidden], dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) # [32, 2, 1]

        multi_hidden2 = torch.stack([audio_hidden, video_hidden], dim=2) # [32, 128, 2]
        fused_feat = torch.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze(-1) #  [32, 128]
        
        emos_out_v  = self.fc_out_v(video_hidden)
        emos_out_a  = self.fc_out_a(audio_hidden)
        emos_out_b  = self.fc_out_b(fused_feat)
        
        if train:
            loss_v = self.emo_loss(emos_out_v, emotions)
            loss_a = self.emo_loss(emos_out_a, emotions)
            loss_b = self.emo_loss(emos_out_b, emotions)                        
            return emos_out_v, emos_out_a, emos_out_b, loss_v, loss_a, loss_b
        else:
            return emos_out_v, emos_out_a, emos_out_b, 0, 0, 0