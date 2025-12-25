"""
Configuration file for DCCMA-Net fake news detection.
"""

import torch

class Config:
    """Configuration class for DCCMA-Net."""
    
    # Data paths
    DATA_DIR = "dataset/subset"
    IMAGE_DIR = "dataset/image_data"
    TRAIN_FILE = "multimodal_train.tsv"
    VAL_FILE = "multimodal_validate.tsv"
    TEST_FILE = "multimodal_test_public.tsv"
    
    # Model paths
    MODEL_SAVE_DIR = "checkpoints"
    LOG_DIR = "logs"
    
    # Model architecture
    TEXT_ENCODER = "bert-base-uncased"  # Can also use "roberta-base"
    IMAGE_ENCODER = "resnet50"           # Can also use "vgg19"
    
    # Text encoding
    MAX_TEXT_LENGTH = 128
    TEXT_EMBED_DIM = 768  # BERT base embedding dimension
    
    # Image encoding
    IMAGE_SIZE = 224
    IMAGE_EMBED_DIM = 2048  # ResNet-50 output dimension (before FC layer)
    
    # Disentanglement
    SHARED_DIM = 512
    SPECIFIC_DIM = 256
    
    # Fusion
    FUSION_DIM = 1024
    HIDDEN_DIM = 512
    NUM_CLASSES = 2  # Binary classification: fake (1) or real (0)
    
    # Training
    BATCH_SIZE = 16      # Small batch size for CPU
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 20
    
    # Optimization
    GRADIENT_CLIP = 1.0
    PATIENCE = 5  # Early stopping patience
    
    # Loss weights
    CLASSIFICATION_WEIGHT = 1.0
    DISENTANGLEMENT_WEIGHT = 0.1  # Weight for disentanglement loss
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading
    NUM_WORKERS = 0  # Set to 0 for Windows compatibility
    PIN_MEMORY = False  # Set to False when using CPU
    
    # Logging
    PRINT_FREQ = 10
    SAVE_FREQ = 1  # Save checkpoint every N epochs
    
    # Reproducibility
    SEED = 42
    
    def __repr__(self):
        """Print configuration."""
        config_str = "\n" + "=" * 60 + "\n"
        config_str += "DCCMA-Net Configuration\n"
        config_str += "=" * 60 + "\n"
        
        for key, value in self.__class__.__dict__.items():
            if not key.startswith('_') and not callable(value):
                config_str += f"{key}: {value}\n"
        
        config_str += "=" * 60 + "\n"
        return config_str

# Create a global config instance
config = Config()

if __name__ == "__main__":
    print(config)
