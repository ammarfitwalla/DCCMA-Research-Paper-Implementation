"""
Training script for DCCMA-Net.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random

from config import config
from models.dccma.dccma_net import DCCMANet
from utils.dataset import create_dataloaders
from utils.metrics import MetricsCalculator, AverageMeter


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device, config, epoch):
    """
    Train for one epoch.
    
    Args:
        model: DCCMA-Net model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        config: Configuration object
        epoch: Current epoch number
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    dis_loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids, attention_mask, images)
        
        # Classification loss
        cls_loss = criterion(logits, labels)
        
        # Disentanglement loss
        with torch.no_grad():
            text_embed = model.text_encoder(input_ids, attention_mask)
            image_embed = model.image_encoder(images)
        
        dis_losses = model.compute_disentanglement_loss(text_embed, image_embed)
        dis_loss = dis_losses['total_disentanglement']
        
        # Total loss
        total_loss = (config.CLASSIFICATION_WEIGHT * cls_loss + 
                     config.DISENTANGLEMENT_WEIGHT * dis_loss)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if config.GRADIENT_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        
        optimizer.step()
        
        # Update metrics
        batch_size = input_ids.size(0)
        loss_meter.update(total_loss.item(), batch_size)
        cls_loss_meter.update(cls_loss.item(), batch_size)
        dis_loss_meter.update(dis_loss.item(), batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'cls': f'{cls_loss_meter.avg:.4f}',
            'dis': f'{dis_loss_meter.avg:.4f}'
        })
    
    return loss_meter.avg, cls_loss_meter.avg, dis_loss_meter.avg


def validate(model, dataloader, criterion, device, config):
    """
    Validate the model.
    
    Args:
        model: DCCMA-Net model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        config: Configuration object
    
    Returns:
        Average loss and metrics dictionary
    """
    model.eval()
    
    loss_meter = AverageMeter()
    metrics_calc = MetricsCalculator(num_classes=config.NUM_CLASSES)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask, images)
            
            # Loss
            loss = criterion(logits, labels)
            
            # Update metrics
            batch_size = input_ids.size(0)
            loss_meter.update(loss.item(), batch_size)
            
            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Update metrics calculator
            metrics_calc.update(preds, labels, probs)
    
    # Compute final metrics
    metrics = metrics_calc.compute()
    
    return loss_meter.avg, metrics


def train(model, train_loader, val_loader, criterion, optimizer, 
          device, config, start_epoch=0):
    """
    Full training loop.
    
    Args:
        model: DCCMA-Net model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        config: Configuration object
        start_epoch: Starting epoch (for resuming)
    
    Returns:
        Training history
    """
    # Setup logging
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    writer = SummaryWriter(config.LOG_DIR)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    best_f1 = 0.0
    patience_counter = 0
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Num epochs: {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print("=" * 60 + "\n")
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_cls_loss, train_dis_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch + 1
        )
        
        print(f"Train Loss: {train_loss:.4f} "
              f"(cls: {train_cls_loss:.4f}, dis: {train_dis_loss:.4f})")
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, config)
        
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('F1/val', val_metrics['f1'], epoch)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_FREQ == 0:
            checkpoint_path = os.path.join(
                config.MODEL_SAVE_DIR, 
                f'dccma_epoch_{epoch + 1}.pt'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            
            best_model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, best_model_path)
            print(f"New best model! F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    writer.close()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best F1 Score: {best_f1:.4f}")
    print("=" * 60 + "\n")
    
    return history


def main():
    """Main training function."""
    
    # Set seed for reproducibility
    set_seed(config.SEED)
    
    # Device
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Create model
    print("\nCreating DCCMA-Net model...")
    model = DCCMANet(
        text_encoder_name=config.TEXT_ENCODER,
        image_encoder_name=config.IMAGE_ENCODER,
        text_embed_dim=config.TEXT_EMBED_DIM,
        image_embed_dim=config.IMAGE_EMBED_DIM,
        shared_dim=config.SHARED_DIM,
        specific_dim=config.SPECIFIC_DIM,
        fusion_dim=config.FUSION_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES
    )
    
    model = model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    tokenizer = model.get_tokenizer()
    train_loader, val_loader, test_loader = create_dataloaders(config, tokenizer)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Train
    history = train(
        model, train_loader, val_loader,
        criterion, optimizer, device, config
    )
    
    # Test on best model
    print("\nEvaluating on test set...")
    best_model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.pt')
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_metrics = validate(model, test_loader, criterion, device, config)
        
        print("\n" + "=" * 60)
        print("TEST SET RESULTS")
        print("=" * 60)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
