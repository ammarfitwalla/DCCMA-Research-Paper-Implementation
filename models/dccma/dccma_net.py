"""
Complete DCCMA-Net model.
Integrates all components for multimodal fake news detection.
"""

import torch
import torch.nn as nn

from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder
from .disentanglement import DisentanglementModule, orthogonality_loss, similarity_loss
from .fusion import FusionModule


class DCCMANet(nn.Module):
    """
    DCCMA-Net: Disentanglement-based Cross-Modal Clues Mining and Aggregation Network.
    
    This model performs multimodal fake news detection by:
    1. Encoding text and images
    2. Disentangling shared and specific features
    3. Mining cross-modal clues via attention
    4. Fusing features for classification
    
    Args:
        text_encoder_name: Pretrained text model name (default: 'bert-base-uncased')
        image_encoder_name: Pretrained image model name (default: 'resnet50')
        text_embed_dim: Text embedding dimension (default: 768)
        image_embed_dim: Image embedding dimension (default: 2048)
        shared_dim: Shared feature dimension (default: 512)
        specific_dim: Specific feature dimension (default: 256)
        fusion_dim: Fusion layer dimension (default: 1024)
        hidden_dim: Hidden dimension for attention (default: 512)
        num_classes: Number of output classes (default: 2)
        freeze_text: Whether to freeze text encoder (default: False)
        freeze_image: Whether to freeze image encoder (default: False)
    """
    
    def __init__(self,
                 text_encoder_name='bert-base-uncased',
                 image_encoder_name='resnet50',
                 text_embed_dim=768,
                 image_embed_dim=2048,
                 shared_dim=512,
                 specific_dim=256,
                 fusion_dim=1024,
                 hidden_dim=512,
                 num_classes=2,
                 freeze_text=False,
                 freeze_image=False):
        super(DCCMANet, self).__init__()
        
        # Text encoder
        self.text_encoder = TextEncoder(
            model_name=text_encoder_name,
            embed_dim=text_embed_dim,
            freeze_bert=freeze_text
        )
        
        # Image encoder
        self.image_encoder = ImageEncoder(
            model_name=image_encoder_name,
            embed_dim=image_embed_dim,
            freeze_backbone=freeze_image
        )
        
        # Text disentanglement
        self.text_disentangler = DisentanglementModule(
            input_dim=text_embed_dim,
            shared_dim=shared_dim,
            specific_dim=specific_dim
        )
        
        # Image disentanglement
        self.image_disentangler = DisentanglementModule(
            input_dim=image_embed_dim,
            shared_dim=shared_dim,
            specific_dim=specific_dim
        )
        
        # Fusion module
        self.fusion = FusionModule(
            text_shared_dim=shared_dim,
            text_specific_dim=specific_dim,
            image_shared_dim=shared_dim,
            image_specific_dim=specific_dim,
            fusion_dim=fusion_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
    
    def forward(self, input_ids, attention_mask, images, return_features=False):
        """
        Forward pass.
        
        Args:
            input_ids: Text token IDs (batch_size, seq_len)
            attention_mask: Text attention mask (batch_size, seq_len)
            images: Image tensors (batch_size, 3, H, W)
            return_features: Whether to return intermediate features
        
        Returns:
            logits: Classification logits (batch_size, num_classes)
            features: Dictionary of intermediate features (if return_features=True)
        """
        # Encode text and images
        text_embed = self.text_encoder(input_ids, attention_mask)
        image_embed = self.image_encoder(images)
        
        # Disentangle features
        text_shared, text_specific = self.text_disentangler(text_embed)
        image_shared, image_specific = self.image_disentangler(image_embed)
        
        # Fusion and classification
        logits, attn_weights = self.fusion(
            text_shared, text_specific,
            image_shared, image_specific
        )
        
        if return_features:
            features = {
                'text_embed': text_embed,
                'image_embed': image_embed,
                'text_shared': text_shared,
                'text_specific': text_specific,
                'image_shared': image_shared,
                'image_specific': image_specific,
                'attention_weights': attn_weights
            }
            return logits, features
        
        return logits
    
    def compute_disentanglement_loss(self, text_embed, image_embed):
        """
        Compute disentanglement losses.
        
        Args:
            text_embed: Text embeddings (batch_size, text_embed_dim)
            image_embed: Image embeddings (batch_size, image_embed_dim)
        
        Returns:
            Dictionary of disentanglement losses
        """
        # Disentangle features
        text_shared, text_specific = self.text_disentangler(text_embed)
        image_shared, image_specific = self.image_disentangler(image_embed)
        
        # Orthogonality loss: shared and specific should be independent
        text_ortho_loss = orthogonality_loss(text_shared, text_specific)
        image_ortho_loss = orthogonality_loss(image_shared, image_specific)
        
        # Similarity loss: shared features across modalities should be similar
        shared_sim_loss = similarity_loss(text_shared, image_shared)
        
        losses = {
            'text_orthogonality': text_ortho_loss,
            'image_orthogonality': image_ortho_loss,
            'shared_similarity': shared_sim_loss,
            'total_disentanglement': text_ortho_loss + image_ortho_loss + shared_sim_loss
        }
        
        return losses
    
    def get_tokenizer(self):
        """Get the text tokenizer."""
        return self.text_encoder.get_tokenizer()


if __name__ == "__main__":
    # Test the complete DCCMA-Net
    print("Testing DCCMA-Net...")
    
    # Create model
    model = DCCMANet(
        text_encoder_name='bert-base-uncased',
        image_encoder_name='resnet50',
        text_embed_dim=768,
        image_embed_dim=2048,
        shared_dim=512,
        specific_dim=256,
        fusion_dim=1024,
        hidden_dim=512,
        num_classes=2
    )
    
    print(f"Model created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    images = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\nInput shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  images: {images.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits, features = model(input_ids, attention_mask, images, return_features=True)
    
    print(f"\nOutput shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  text_embed: {features['text_embed'].shape}")
    print(f"  image_embed: {features['image_embed'].shape}")
    print(f"  text_shared: {features['text_shared'].shape}")
    print(f"  text_specific: {features['text_specific'].shape}")
    print(f"  image_shared: {features['image_shared'].shape}")
    print(f"  image_specific: {features['image_specific'].shape}")
    
    # Test disentanglement losses
    with torch.no_grad():
        losses = model.compute_disentanglement_loss(
            features['text_embed'],
            features['image_embed']
        )
    
    print(f"\nDisentanglement losses:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
    
    print("\nDCCMA-Net test passed!")
