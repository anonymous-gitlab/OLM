import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ResNet18 
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, Attention, Bilinear

class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()
        fusion_type = args.fusion_method
        n_classes = self.get_num_classes(args.dataset)
        
        self.fusion_module = self.create_fusion_module(fusion_type, n_classes)
        self.audio_net = ResNet18(modality='audio')
        self.visual_net = ResNet18(modality='visual')
        self.register_buffer('initial_loss', torch.tensor([1.0, 1.0, 1.0]))
        self.register_buffer('train_initial_loss', torch.tensor([1.0, 1.0, 1.0]))
        self.register_buffer('eval_initial_loss', torch.tensor([1.0, 1.0, 1.0]))

    def get_num_classes(self, dataset_name):
        # Define the number of classes based on the dataset name
        class_mapping = {
            'VGGSound': 309,
            'KineticSound': 31,
            'CREMAD': 6,
            'AVE': 28
        }
        if dataset_name in class_mapping:
            return class_mapping[dataset_name]
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(dataset_name))

    def create_fusion_module(self, fusion_type, output_dim):
        # Create the fusion module based on the fusion type
        if fusion_type == 'sum':
            return SumFusion(output_dim=output_dim)
        elif fusion_type == 'concat':
            return ConcatFusion(output_dim=output_dim)
        elif fusion_type == 'film':
            return FiLM(output_dim=output_dim, x_film=True)
        elif fusion_type == 'gated':
            return GatedFusion(output_dim=output_dim, x_gate=True)
        elif fusion_type == 'attention':
            return Attention(output_dim=output_dim)
        elif fusion_type == 'bilinear':
            return Bilinear(output_dim=output_dim)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion_type))

    def forward(self, audio, visual, label, train):
        # Process audio and visual inputs
        audio_features = self.audio_net(audio)
        visual_features = self.visual_net(visual)

        # Reshape visual features for fusion
        (_, C, H, W) = visual_features.size()
        B = audio_features.size()[0]
        visual_features = visual_features.view(B, -1, C, H, W)
        visual_features = visual_features.permute(0, 2, 1, 3, 4)

        # Apply adaptive pooling
        audio_features = F.adaptive_avg_pool2d(audio_features, 1)
        visual_features = F.adaptive_avg_pool3d(visual_features, 1)

        # Flatten audio and visual features
        audio_features = torch.flatten(audio_features, 1)
        visual_features = torch.flatten(visual_features, 1)

        # Apply fusion module
        audio_output, visual_output, final_output = self.fusion_module(audio_features, visual_features)

        return audio_output, visual_output, final_output
