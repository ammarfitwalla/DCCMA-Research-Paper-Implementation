"""
Image encoder for DCCMA-Net.
Uses ResNet or VGG for image encoding.
"""

import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    """
    Image encoder using pretrained ResNet or VGG.
    
    Args:
        model_name: Name of the pretrained model ('resnet50', 'resnet101', 'vgg19')
        embed_dim: Dimension of the output embedding
        freeze_backbone: Whether to freeze backbone parameters
    """
    
    def __init__(self, model_name='resnet50', embed_dim=2048, freeze_backbone=False):
        super(ImageEncoder, self).__init__()
        
        self.model_name = model_name
        self.embed_dim = embed_dim
        
        # Load pretrained model
        if model_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            self.feature_dim = 2048
            # Remove the final FC layer
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
            
        elif model_name == 'resnet101':
            backbone = models.resnet101(pretrained=True)
            self.feature_dim = 2048
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
            
        elif model_name == 'vgg19':
            backbone = models.vgg19(pretrained=True)
            self.feature_dim = 512 * 7 * 7  # VGG19 features before avgpool
            self.encoder = backbone.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Projection layer if embed_dim != feature_dim
        if self.feature_dim != embed_dim:
            self.projection = nn.Linear(self.feature_dim, embed_dim)
        else:
            self.projection = nn.Identity()
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, images):
        """
        Forward pass.
        
        Args:
            images: Image tensor (batch_size, 3, H, W)
        
        Returns:
            Image embeddings (batch_size, embed_dim)
        """
        # Extract features
        features = self.encoder(images)  # (batch_size, feature_dim, H', W')
        
        # Flatten features
        if 'vgg' in self.model_name:
            features = self.avgpool(features)
        
        features = features.view(features.size(0), -1)  # (batch_size, feature_dim)
        
        # Project to desired embedding dimension
        image_embed = self.projection(features)  # (batch_size, embed_dim)
        
        return image_embed


if __name__ == "__main__":
    # Test the image encoder
    print("Testing ImageEncoder...")
    
    encoder = ImageEncoder(model_name='resnet50', embed_dim=2048)
    print(f"Model: {encoder.model_name}")
    print(f"Feature dim: {encoder.feature_dim}")
    print(f"Embed dim: {encoder.embed_dim}")
    
    # Create dummy images
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    print(f"\nInput images shape: {images.shape}")
    
    # Test forward pass
    with torch.no_grad():
        embeddings = encoder(images)
    
    print(f"Output embeddings shape: {embeddings.shape}")
    print("\nImageEncoder test passed!")
