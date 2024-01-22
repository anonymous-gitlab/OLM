import math
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(Attention, self).__init__()
        
        self.o_x = nn.Linear(input_dim, output_dim)
        self.o_y = nn.Linear(input_dim, output_dim)
        self.attention_mlp = nn.Linear(2*input_dim, input_dim)
        self.fc_att =  nn.Linear(input_dim, 2)
        self.o_union = nn.Linear(input_dim, output_dim)
    
    def forward(self, x, y):
        audio_hidden = x
        video_hidden = y    
        multi_hidden1 = torch.cat([audio_hidden, video_hidden], dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        score = self.fc_att(attention)
        score = torch.unsqueeze(score, 2) # [32, 2, 1]
        
        multi_hidden2 = torch.stack([audio_hidden, video_hidden], dim=2) # [32, 128, 2]
        fused_feat = torch.matmul(multi_hidden2, score)
        fused_feat = fused_feat.squeeze(-1) #  [32, 128]        

        audio_logits = self.o_x(x)
        visual_logits = self.o_y(y)
        pivot_logits = self.o_union(fused_feat)
                
        return audio_logits, visual_logits, pivot_logits
    
class BilinearFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(BilinearFusion, self).__init__()
        embed_dim = input_dim
        self.weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.classifier = nn.Linear(1, output_dim)        
        self.o_x = nn.Linear(input_dim, output_dim)
        self.o_y = nn.Linear(input_dim, output_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))        
        # nn.init.xavier_uniform_(self.weight)
                
    def forward(self, audio_tensor, video_tensor,):
        # or compute bilinear fusion like this
        fused_tensor = torch.einsum('bi,ij,bj->b', video_tensor, self.weight, audio_tensor)
        # fused_tensor = fused_tensor.squeeze(-1)        
        fused_tensor = fused_tensor.view(fused_tensor.size(0), -1)  # Reshape to (batch_size, )
        video_logits = self.o_x(video_tensor)
        audio_logits = self.o_y(audio_tensor)        
        pivot_logits = self.classifier(fused_tensor)
        return audio_logits, video_logits, pivot_logits


class Bilinear(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(Bilinear, self).__init__()
        embed_dim = input_dim
        self.tucker_a = nn.Linear(embed_dim, embed_dim)
        self.tucker_v = nn.Linear(embed_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, output_dim)        
        self.o_x = nn.Linear(embed_dim, output_dim)
        self.o_y = nn.Linear(embed_dim, output_dim)

    def forward(self, audio_tensor, video_tensor,):        
        video_embed = self.tucker_v(video_tensor)
        audio_embed = self.tucker_a(audio_tensor)        
        # element_product 
        fused_output = video_embed * audio_embed        
        video_logits = self.o_x(video_tensor)
        audio_logits = self.o_y(audio_tensor)        
        pivot_logits = self.classifier(fused_output)
                                
        return audio_logits, video_logits, pivot_logits


class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)
        self.o_x = nn.Linear(input_dim, output_dim)
        self.o_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        x = self.o_x(x)
        y = self.o_y(y)
        
        return x, y, output


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(2*input_dim, output_dim)
        self.o_x = nn.Linear(input_dim, output_dim)
        self.o_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        x = self.o_x(x)
        y = self.o_y(y)                
        return x, y, output


class FiLM(nn.Module):

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)
        self.o_x = nn.Linear(input_dim, output_dim)
        self.o_y = nn.Linear(input_dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)
        
        x = self.o_x(x)
        y = self.o_y(y)        

        return x, y, output


class GatedFusion(nn.Module):

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)
        self.o_x = nn.Linear(input_dim, output_dim)
        self.o_y = nn.Linear(input_dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))
        x = self.o_x(x)
        y = self.o_y(y) 
             
        return x, y, output
        # return out_x, out_y, output