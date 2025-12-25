"""
Disentanglement module for DCCMA-Net.
Separates shared and modality-specific features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DisentanglementModule(nn.Module):
    """
    Disentanglement module to separate shared and specific features.
    
    Based on the DCCMA paper, this module learns to separate:
    - Shared features: Common information between text and image modalities
    - Specific features: Unique information for each modality
    
    Args:
        input_dim: Input embedding dimension (from encoders)
        shared_dim: Dimension of shared features
        specific_dim: Dimension of modality-specific features
    """
    
    def __init__(self, input_dim, shared_dim=512, specific_dim=256):
        super(DisentanglementModule, self).__init__()
        
        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.specific_dim = specific_dim
        
        # Shared feature extractor
        self.shared_extractor = nn.Sequential(
            nn.Linear(input_dim, shared_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(shared_dim * 2, shared_dim),
            nn.ReLU()
        )
        
        # Specific feature extractor
        self.specific_extractor = nn.Sequential(
            nn.Linear(input_dim, specific_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(specific_dim * 2, specific_dim),
            nn.ReLU()
        )
    
    def forward(self, features):
        """
        Extract shared and specific features.
        
        Args:
            features: Input features (batch_size, input_dim)
        
        Returns:
            shared_features: Shared features (batch_size, shared_dim)
            specific_features: Specific features (batch_size, specific_dim)
        """
        shared = self.shared_extractor(features)
        specific = self.specific_extractor(features)
        
        return shared, specific


def orthogonality_loss(features1, features2):
    """
    Compute orthogonality loss to ensure features are independent.
    
    This loss encourages the two feature sets to be orthogonal,
    meaning they capture different information.
    
    Works with features of different dimensions by projecting to common space.
    
    Args:
        features1: First feature set (batch_size, dim1)
        features2: Second feature set (batch_size, dim2)
    
    Returns:
        Orthogonality loss (scalar)
    """
    # Normalize features
    features1_norm = F.normalize(features1, p=2, dim=1)
    features2_norm = F.normalize(features2, p=2, dim=1)
    
    # For features of different dimensions, pad the smaller one
    dim1 = features1_norm.size(1)
    dim2 = features2_norm.size(1)
    
    if dim1 != dim2:
        # Pad to match dimensions
        max_dim = max(dim1, dim2)
        if dim1 < max_dim:
            padding = torch.zeros(features1_norm.size(0), max_dim - dim1, 
                                device=features1_norm.device)
            features1_norm = torch.cat([features1_norm, padding], dim=1)
        if dim2 < max_dim:
            padding = torch.zeros(features2_norm.size(0), max_dim - dim2,
                                device=features2_norm.device)
            features2_norm = torch.cat([features2_norm, padding], dim=1)
    
    # Compute element-wise product and sum (measures correlation)
    # We want this to be close to 0 (orthogonal)
    correlation = torch.sum(features1_norm * features2_norm, dim=1)
    
    # Mean absolute correlation
    loss = torch.mean(torch.abs(correlation))
    
    return loss


def similarity_loss(features1, features2):
    """
    Compute similarity loss to encourage features to be similar.
    
    This is used for shared features across modalities - we want
    the shared features from text and image to be similar.
    
    Args:
        features1: First feature set (batch_size, dim)
        features2: Second feature set (batch_size, dim)
    
    Returns:
        Similarity loss (scalar)
    """
    # Normalize features
    features1_norm = F.normalize(features1, p=2, dim=1)
    features2_norm = F.normalize(features2, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(features1_norm, features2_norm, dim=1)
    
    # We want similarity to be high (close to 1), so minimize 1 - similarity
    loss = torch.mean(1 - similarity)
    
    return loss


if __name__ == "__main__":
    # Test the disentanglement module
    print("Testing DisentanglementModule...")
    
    batch_size = 4
    input_dim = 768
    shared_dim = 512
    specific_dim = 256
    
    # Create module
    disentangler = DisentanglementModule(
        input_dim=input_dim,
        shared_dim=shared_dim,
        specific_dim=specific_dim
    )
    
    print(f"Input dim: {input_dim}")
    print(f"Shared dim: {shared_dim}")
    print(f"Specific dim: {specific_dim}")
    
    # Create dummy features for text and image
    text_features = torch.randn(batch_size, input_dim)
    image_features = torch.randn(batch_size, input_dim)
    
    print(f"\nText features shape: {text_features.shape}")
    print(f"Image features shape: {image_features.shape}")
    
    # Extract disentangled features
    with torch.no_grad():
        text_shared, text_specific = disentangler(text_features)
        image_shared, image_specific = disentangler(image_features)
    
    print(f"\nText shared shape: {text_shared.shape}")
    print(f"Text specific shape: {text_specific.shape}")
    print(f"Image shared shape: {image_shared.shape}")
    print(f"Image specific shape: {image_specific.shape}")
    
    # Test losses
    with torch.no_grad():
        ortho_loss = orthogonality_loss(text_shared, text_specific)
        sim_loss = similarity_loss(text_shared, image_shared)
    
    print(f"\nOrthogonality loss: {ortho_loss.item():.4f}")
    print(f"Similarity loss: {sim_loss.item():.4f}")
    
    print("\nDisentanglementModule test passed!")
