"""
Cross-modal fusion module for DCCMA-Net.
Implements cross-modal attention and feature fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism.
    
    Allows one modality to attend to another modality's features.
    This helps capture cross-modal clues and inconsistencies.
    
    Args:
        query_dim: Dimension of query features
        key_dim: Dimension of key features
        hidden_dim: Dimension of attention hidden layer
    """
    
    def __init__(self, query_dim, key_dim, hidden_dim=512):
        super(CrossModalAttention, self).__init__()
        
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        
        self.scale = hidden_dim ** -0.5
    
    def forward(self, query, key, value):
        """
        Compute cross-modal attention.
        
        Args:
            query: Query features from one modality (batch_size, query_dim)
            key: Key features from another modality (batch_size, key_dim)
            value: Value features from another modality (batch_size, key_dim)
        
        Returns:
            Attended features (batch_size, hidden_dim)
            Attention weights (batch_size, 1)
        """
        # Project to hidden dimension
        Q = self.query_proj(query)  # (batch_size, hidden_dim)
        K = self.key_proj(key)      # (batch_size, hidden_dim)
        V = self.value_proj(value)  # (batch_size, hidden_dim)
        
        # Compute attention scores
        # For simplicity, using dot product attention
        scores = torch.sum(Q * K, dim=1, keepdim=True) * self.scale  # (batch_size, 1)
        attn_weights = torch.sigmoid(scores)  # (batch_size, 1)
        
        # Apply attention to values
        attended = attn_weights * V  # (batch_size, hidden_dim)
        
        return attended, attn_weights


class FusionModule(nn.Module):
    """
    Feature fusion module for DCCMA-Net.
    
    Fuses shared, specific, and cross-modal features for final classification.
    
    Args:
        text_shared_dim: Dimension of text shared features
        text_specific_dim: Dimension of text specific features
        image_shared_dim: Dimension of image shared features
        image_specific_dim: Dimension of image specific features
        fusion_dim: Dimension of fused features
        hidden_dim: Hidden dimension for cross-modal attention
        num_classes: Number of output classes (2 for binary)
    """
    
    def __init__(self, text_shared_dim, text_specific_dim, 
                 image_shared_dim, image_specific_dim,
                 fusion_dim=1024, hidden_dim=512, num_classes=2):
        super(FusionModule, self).__init__()
        
        # Cross-modal attention
        # Text attends to image
        self.text_to_image_attn = CrossModalAttention(
            query_dim=text_shared_dim,
            key_dim=image_shared_dim,
            hidden_dim=hidden_dim
        )
        
        # Image attends to text
        self.image_to_text_attn = CrossModalAttention(
            query_dim=image_shared_dim,
            key_dim=text_shared_dim,
            hidden_dim=hidden_dim
        )
        
        # Calculate total feature dimension
        total_dim = (text_shared_dim + text_specific_dim + 
                    image_shared_dim + image_specific_dim + 
                    2 * hidden_dim)  # +2*hidden_dim for cross-modal attended features
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)
    
    def forward(self, text_shared, text_specific, image_shared, image_specific):
        """
        Fuse multimodal features and classify.
        
        Args:
            text_shared: Shared text features (batch_size, text_shared_dim)
            text_specific: Specific text features (batch_size, text_specific_dim)
            image_shared: Shared image features (batch_size, image_shared_dim)
            image_specific: Specific image features (batch_size, image_specific_dim)
        
        Returns:
            logits: Classification logits (batch_size, num_classes)
            attention_weights: Dictionary of attention weights for visualization
        """
        # Cross-modal attention
        # Text queries image
        text_attend_image, t2i_weights = self.text_to_image_attn(
            text_shared, image_shared, image_shared
        )
        
        # Image queries text
        image_attend_text, i2t_weights = self.image_to_text_attn(
            image_shared, text_shared, text_shared
        )
        
        # Concatenate all features
        fused_features = torch.cat([
            text_shared,
            text_specific,
            image_shared,
            image_specific,
            text_attend_image,
            image_attend_text
        ], dim=1)
        
        # Fusion and classification
        fused = self.fusion(fused_features)
        logits = self.classifier(fused)
        
        # Store attention weights for analysis
        attn_weights = {
            'text_to_image': t2i_weights,
            'image_to_text': i2t_weights
        }
        
        return logits, attn_weights


if __name__ == "__main__":
    # Test the fusion module
    print("Testing FusionModule...")
    
    batch_size = 4
    text_shared_dim = 512
    text_specific_dim = 256
    image_shared_dim = 512
    image_specific_dim = 256
    fusion_dim = 1024
    hidden_dim = 512
    num_classes = 2
    
    # Create module
    fusion = FusionModule(
        text_shared_dim=text_shared_dim,
        text_specific_dim=text_specific_dim,
        image_shared_dim=image_shared_dim,
        image_specific_dim=image_specific_dim,
        fusion_dim=fusion_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    
    print(f"Text shared dim: {text_shared_dim}")
    print(f"Text specific dim: {text_specific_dim}")
    print(f"Image shared dim: {image_shared_dim}")
    print(f"Image specific dim: {image_specific_dim}")
    
    # Create dummy features
    text_shared = torch.randn(batch_size, text_shared_dim)
    text_specific = torch.randn(batch_size, text_specific_dim)
    image_shared = torch.randn(batch_size, image_shared_dim)
    image_specific = torch.randn(batch_size, image_specific_dim)
    
    # Forward pass
    with torch.no_grad():
        logits, attn_weights = fusion(text_shared, text_specific, image_shared, image_specific)
    
    print(f"\nLogits shape: {logits.shape}")
    print(f"Text-to-image attention shape: {attn_weights['text_to_image'].shape}")
    print(f"Image-to-text attention shape: {attn_weights['image_to_text'].shape}")
    
    print("\nFusionModule test passed!")
