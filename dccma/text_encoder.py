"""
Text encoder for DCCMA-Net.
Uses BERT or RoBERTa for text encoding.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer


class TextEncoder(nn.Module):
    """
    Text encoder using pretrained BERT/RoBERTa.
    
    Args:
        model_name: Name of the pretrained model ('bert-base-uncased' or 'roberta-base')
        embed_dim: Dimension of the output embedding
        freeze_bert: Whether to freeze BERT parameters
    """
    
    def __init__(self, model_name='bert-base-uncased', embed_dim=768, freeze_bert=False):
        super(TextEncoder, self).__init__()
        
        self.model_name = model_name
        self.embed_dim = embed_dim
        
        # Load pretrained model
        if 'roberta' in model_name.lower():
            self.encoder = RobertaModel.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        else:
            self.encoder = BertModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Get hidden size from model config
        self.hidden_size = self.encoder.config.hidden_size
        
        # Projection layer if embed_dim != hidden_size
        if self.hidden_size != embed_dim:
            self.projection = nn.Linear(self.hidden_size, embed_dim)
        else:
            self.projection = nn.Identity()
        
        # Optionally freeze BERT
        if freeze_bert:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            Text embeddings (batch_size, embed_dim)
        """
        # Get BERT outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Project to desired embedding dimension
        text_embed = self.projection(cls_output)  # (batch_size, embed_dim)
        
        return text_embed
    
    def get_tokenizer(self):
        """Return the tokenizer for this encoder."""
        return self.tokenizer


if __name__ == "__main__":
    # Test the text encoder
    print("Testing TextEncoder...")
    
    encoder = TextEncoder(model_name='bert-base-uncased', embed_dim=768)
    print(f"Model: {encoder.model_name}")
    print(f"Hidden size: {encoder.hidden_size}")
    print(f"Embed dim: {encoder.embed_dim}")
    
    # Test tokenization
    tokenizer = encoder.get_tokenizer()
    texts = ["This is a fake news article.", "This is real news."]
    
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    print(f"\nInput IDs shape: {encoded['input_ids'].shape}")
    print(f"Attention mask shape: {encoded['attention_mask'].shape}")
    
    # Test forward pass
    with torch.no_grad():
        embeddings = encoder(encoded['input_ids'], encoded['attention_mask'])
    
    print(f"Output embeddings shape: {embeddings.shape}")
    print("\nTextEncoder test passed!")
