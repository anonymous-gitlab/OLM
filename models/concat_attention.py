import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, audio_dim, text_dim, video_dim, output_dim1, output_dim2=1, layers='256,128', dropout=0.3):
        super(Attention, self).__init__()

        self.audio_mlp = self.MLP(audio_dim, layers, dropout)
        self.text_mlp  = self.MLP(text_dim,  layers, dropout)
        self.video_mlp = self.MLP(video_dim, layers, dropout)

        layers_list = list(map(lambda x: int(x), layers.split(',')))
        hiddendim = layers_list[-1] * 3
        self.attention_mlp = self.MLP(hiddendim, layers, dropout)

        self.fc_att   = nn.Linear(layers_list[-1], 3)
        self.fc_out_1 = nn.Linear(layers_list[-1], output_dim1)
        self.fc_out_2 = nn.Linear(layers_list[-1], output_dim2)
    
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

    def forward(self, audio_feat, text_feat, video_feat):
        audio_hidden = self.audio_mlp(audio_feat) # [32, 128] [32,1024]
        text_hidden  = self.text_mlp(text_feat)   # [32, 128] [32,1024]
        video_hidden = self.video_mlp(video_feat) # [32, 128] [32,1024]

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]

        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2) # [32, 128, 3]
        fused_feat = torch.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze() # [32, 128]
        emos_out  = self.fc_out_1(fused_feat)
        vals_out  = self.fc_out_2(fused_feat)
        return fused_feat, emos_out, vals_out