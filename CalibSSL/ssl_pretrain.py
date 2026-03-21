import torch
import torch.nn as nn
from tqdm import tqdm

def pretrain_vime(X_unlabeled, input_dim, hidden_dim=256, epochs=50, 
                  mask_prob=0.3, batch_size=256, lr=1e-3, device='cuda'):
    """
    Pretrain ViME encoder using self-supervised learning
    
    Args:
        X_unlabeled: Unlabeled data (torch.Tensor)
        input_dim: Number of features
        hidden_dim: Hidden dimension size
        epochs: Number of pretraining epochs
        mask_prob: Probability of masking each feature
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda' or 'cpu'
    
    Returns:
        Pretrained encoder (nn.Module)
    """
    from models import ViMEEncoder
    
    # Initialize model
    vime = ViMEEncoder(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(vime.parameters(), lr=lr)
    
    # Move data to device
    X_unlabeled = X_unlabeled.to(device)
    
    # Create dataloader
    dataset = torch.utils.data.TensorDataset(X_unlabeled)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss functions
    recon_criterion = nn.MSELoss()
    mask_criterion = nn.BCELoss()
    
    # Training loop
    vime.train()
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_mask_loss = 0
        
        for (batch_x,) in loader:
            # Generate random mask
            mask = torch.bernoulli(torch.full_like(batch_x, mask_prob))
            
            # Forward pass
            x_recon, mask_pred, _ = vime(batch_x, mask)
            
            # Reconstruction loss (only on masked features)
            recon_loss = recon_criterion(x_recon * mask, batch_x * mask)
            
            # Mask prediction loss
            mask_loss = mask_criterion(mask_pred, mask)
            
            # Combined loss
            loss = recon_loss + mask_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_mask_loss += mask_loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            avg_recon = total_recon_loss / len(loader)
            avg_mask = total_mask_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | Mask: {avg_mask:.4f}")
    
    # Return only the encoder part
    return vime.encoder


# Test
if __name__ == "__main__":
    from data_loader import load_dataset
    
    print("Testing SSL pretraining...")
    X_lab, y_lab, X_unlab, X_test, y_test = load_dataset('adult', label_fraction=0.1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    encoder = pretrain_vime(
        X_unlab, 
        input_dim=X_unlab.shape[1],
        hidden_dim=128,
        epochs=20,
        device=device
    )
    
    print("✅ SSL pretraining completed!")
    print(f"Encoder output shape: {encoder(X_test[:10].to(device)).shape}")