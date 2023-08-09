from __future__ import annotations
from functools import partial

import copy
import math
import torch
import logging
import torch.nn as nn
from torch import Tensor
from functools import reduce
from torch.nn.init import xavier_uniform_

import torch.nn.functional as F
from typing import Optional, Any
from torch.nn.modules import Module
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.container import ModuleList
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.activation import MultiheadAttention
from timm.models.layers import trunc_normal_, DropPath

class EfficientMultimodalTransformer(Module):
    def __init__(self,
                 in_chans: int = 1024,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 start_fusion_layer: int = 2,
                 pivot_num: int = 12,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 output_dim1: int = 6,
                 drop_rate: float = 0.1,
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 norm_layer=None,
                 class_token=True,                 
                 ):
        super(EfficientMultimodalTransformer, self).__init__()
        self.embed_dim = embed_dim = d_model
        self.pivot_size = pivot_size = pivot_num
        self.num_encoder_layers = num_encoder_layers
        self.start_fusion_layer = start_fusion_layer
        self.fusion_layer = num_encoder_layers - start_fusion_layer
        self.cls_token = class_token
        assert self.start_fusion_layer >= 0 and self.start_fusion_layer <= self.num_encoder_layers-1, "check your fusion layer"

        self.proj_v = nn.Linear(in_chans, embed_dim)
        self.proj_a = nn.Linear(in_chans, embed_dim)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        UNI_visual_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.unimodal_vision_layers = _get_clones(UNI_visual_encoder_layer, self.num_encoder_layers)
        UNI_audio_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.unimodal_audio_layers = _get_clones(UNI_audio_encoder_layer, self.num_encoder_layers)
        UNI_fusion_encoder_layer = InterRepFusionBlock(d_model, nhead, dim_feedforward, dropout, activation)
        self.bimodal_fusion_layers = _get_clones(UNI_fusion_encoder_layer, self.num_encoder_layers)
        self.bottleneck = nn.Parameter(data=torch.zeros(pivot_size, 1,  embed_dim))
        self.cls_token_v = nn.Parameter(data=torch.zeros(1, 1, embed_dim)) if class_token else None
        self.cls_token_a = nn.Parameter(data=torch.zeros(1, 1, embed_dim)) if class_token else None
        self.register_buffer("false_column", torch.zeros(1, 1, dtype=torch.bool))
        self.register_buffer('initial_loss', torch.tensor([1.0,1.0,1.0]))        
        self.register_buffer('train_initial_loss', torch.tensor([1.0,1.0,1.0]))        
        self.register_buffer('eval_initial_loss', torch.tensor([1.0,1.0,1.0]))        

        self.fc_out_v = nn.Linear(embed_dim, output_dim1)
        self.fc_out_a = nn.Linear(embed_dim, output_dim1)
        self.fc_out_b = nn.Linear(embed_dim, output_dim1)
        
        self.norm = LayerNorm(d_model)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
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

    def forward(self, src_v, src_key_padding_mask_v, src_a, src_key_padding_mask_a, src_t, src_key_padding_mask_t):        
        """Pass the input through the encoder layers in turn.
        Args:
            src_v: the sequence to the vision encoder (required). # torch.Size([1104, 256, 1024])
            src_a: the sequence to the audio encoder (required).  # torch.Size([2303, 256, 1024])
            src_t: the sequence to the text encoder (required).   # torch.Size([92, 256, 1024])
            mask_v: the mask for the src_v sequence (optional).
            mask_a: the mask for the src_v sequence (optional).
            mask_t: the mask for the src_v sequence (optional).
            src_key_padding_mask_v: the mask for the src_v keys per batch (optional). torch.Size([256, 1104])
            src_key_padding_mask_a: the mask for the src_a keys per batch (optional). torch.Size([256, 2303])
            src_key_padding_mask_t: the mask for the src_t keys per batch (optional). torch.Size([256, 92])
        Shape:
            src_v: (S,N,E), (S,B,E), namely batch_size second
            src_a: (S,N,E), (S,B,E), namely batch_size second
            src_t: (S,N,E), (S,B,E), namely batch_size second
            src_key_padding_mask_v: (N,S), namely batch_size, sequence_length 
            src_key_padding_mask_a: (N,S), namely batch_size, sequence_length
            src_key_padding_mask_t: (N,S), namely batch_size, sequence_length
        """
        batch_size = src_v.shape[1]
        assert all(input.shape[1] == batch_size for input in [src_a, src_t]), "batch size error: check your modality input"
        shared_pivot = self.bottleneck.expand(-1, batch_size, -1)   
        
        src_v = self.proj_v(src_v)
        src_a = self.proj_a(src_a)
        
        if self.cls_token:            
            src_v = torch.cat((self.cls_token_v.expand(-1, batch_size, -1), src_v), dim=0)            
            src_key_padding_mask_v = torch.cat((src_key_padding_mask_v, self.false_column.expand(batch_size, -1)), dim=1)
            src_a = torch.cat((self.cls_token_a.expand(-1, batch_size, -1), src_a), dim=0)
            src_key_padding_mask_a = torch.cat((src_key_padding_mask_a, self.false_column.expand(batch_size, -1)), dim=1)
            
        output_v = src_v
        output_a = src_a
        output_p = shared_pivot

        output_v = self.pos_drop(output_v)
        output_a = self.pos_drop(output_a)
        output_p = self.pos_drop(output_p)

        for mod_v, mod_a, mod_p in zip(self.unimodal_vision_layers,self.unimodal_audio_layers,self.bimodal_fusion_layers):
            output_v = mod_v(output_v, src_mask=None, src_key_padding_mask=src_key_padding_mask_v)
            output_a = mod_a(output_a, src_mask=None, src_key_padding_mask=src_key_padding_mask_a)
            output_p = mod_p(output_v, output_a, output_p,
                             src_v_mask=None, src_a_mask=None,
                             src_v_key_padding_mask=src_key_padding_mask_v,
                             src_a_key_padding_mask=src_key_padding_mask_a)
        
        if self.norm is not None:
            output_v = self.norm(output_v)
            output_a = self.norm(output_a)
            output_p = self.norm(output_p)
        
        output_v, output_a, output_p = output_v.permute(1,0,2), output_a.permute(1,0,2), output_p.permute(1,0,2)
        
        # current version
        # We can also consider the output sequnences [batch, max_len, embedding_dim], the average across max_len dimensions to get averaged/mean embeddings
        # Step 1: Expand Attention/Padding Mask from [batch_size, max_len] to [batch_size, max_len, hidden_size].
        # Step 2: Sum Embeddings along max_len axis so now we have [batch_size, hidden_size].
        # Step 3: Sum Mask along max_len axis. This is done so that we can ignore padding tokens.
        # Step 4: Take Average.
        vision_mask_expanded = (~src_key_padding_mask_v).float().unsqueeze(-1).expand(output_v.size())
        audio_mask_expanded  = (~src_key_padding_mask_a).float().unsqueeze(-1).expand(output_a.size())
        
        vision_sum_embeddings = torch.sum(output_v * vision_mask_expanded, axis=1)
        audio_sum_embeddings = torch.sum(output_a * audio_mask_expanded, axis=1)
        
        vision_sum_mask = vision_mask_expanded.sum(axis=1)
        vision_sum_mask = torch.clamp(vision_sum_mask, min=1e-9)
        audio_sum_mask = audio_mask_expanded.sum(axis=1)
        audio_sum_mask = torch.clamp(audio_sum_mask, min=1e-9)
        
        vision_mean_embeddings = vision_sum_embeddings / vision_sum_mask # torch.Size([32, 768]
        audio_mean_embeddings = audio_sum_embeddings / audio_sum_mask   # torch.Size([32, 768]
        pivot_mean_embeddings = torch.mean(output_p, axis=1)

        emos_out_v = self.fc_out_v(vision_mean_embeddings)
        emos_out_a = self.fc_out_a(audio_mean_embeddings)
        emos_out_b = self.fc_out_b(pivot_mean_embeddings)

        return emos_out_v, emos_out_a, emos_out_b



class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)
        return output


class InterRepFusionBlock(Module):
    r"""InterRepFusionBlock is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(InterRepFusionBlock, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_v = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_a = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.mlp_bimodal_v = Linear(2*d_model, d_model)
        self.mlp_bimodal_a = Linear(2*d_model, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.norm4 = LayerNorm(d_model)
        self.norm5 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(InterRepFusionBlock, self).__setstate__(state)
        
    def forward(self, src_v: Tensor, src_a: Tensor, src_b: Tensor, 
                src_v_mask: Optional[Tensor] = None, 
                src_a_mask: Optional[Tensor] = None, 
                src_v_key_padding_mask: Optional[Tensor] = None,
                src_a_key_padding_mask: Optional[Tensor] = None
                ) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_bv2 = self.multihead_attn_v(src_b, src_v, src_v, attn_mask=src_v_mask,
                                   key_padding_mask=src_v_key_padding_mask)[0]
        src_bv1 = src_b + self.dropout1(src_bv2)
        src_bv1 = self.norm1(src_bv1)

        src_ba2 = self.multihead_attn_a(src_b, src_a, src_a, attn_mask=src_a_mask,
                                   key_padding_mask=src_a_key_padding_mask)[0]
        src_ba1 = src_b + self.dropout2(src_ba2)
        src_ba1 = self.norm2(src_ba1)
        
        gating_v = torch.sigmoid(self.mlp_bimodal_v(torch.cat([src_v[0], src_a[0]], dim=-1)))
        gating_a = torch.sigmoid(self.mlp_bimodal_a(torch.cat([src_a[0], src_v[0]], dim=-1)))
        
        src_bv2 = src_bv1 + src_bv1*gating_v.expand(src_bv1.shape[0],-1,-1) # repeat sequence times
        src_bv2 = self.norm3(src_bv2) # torch.Size([12, 16, 1024])
        
        src_ba2 = src_ba1 + src_ba1*gating_a.expand(src_ba1.shape[0],-1,-1) # repeat sequence times
        src_ba2 = self.norm4(src_ba2) # torch.Size([12, 16, 1024])
        
        src_bav = self.linear2(self.dropout(self.activation(self.linear1(torch.sum(torch.stack([src_bv2,src_ba2], dim=0),dim=0)))))
        src_b = src_bv2 + src_ba2 + self.dropout3(src_bav)
        src_b = self.norm5(src_b)        
        return src_b


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == 'sigmoid':
        return F.sigmoid
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

