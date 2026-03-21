import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def train_supervised(model, X_train, y_train, X_val, y_val, 
                     epochs=100, batch_size=256, lr=1e-3, 
                     device='cuda', patience=10, verbose=True):
    """
    Standard supervised training
    
    Returns:
        Trained model
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(best_state)
    return model


def train_with_calibration_reg(model, X_train, y_train, X_val, y_val,
                                epochs=100, batch_size=256, lr=1e-3,
                                lambda_conf=0.1, device='cuda', patience=10, verbose=True):
    """
    Training with calibration-aware regularization (YOUR NOVEL CONTRIBUTION)
    
    Loss = CrossEntropy + λ * ConfidencePenalty
    
    Confidence Penalty encourages the model to be less overconfident
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            
            # Standard cross-entropy loss
            loss_ce = criterion(logits, batch_y)
            
            # Confidence penalty: encourage higher entropy (less confidence)
            # entropy = -sum(p * log(p))
            epsilon = 1e-8
            entropy = -(probs * torch.log(probs + epsilon)).sum(dim=1).mean()
            
            # We want to MAXIMIZE entropy (minimize negative entropy)
            # So we add negative entropy to loss
            loss_conf = -entropy
            
            # Combined loss
            loss = loss_ce + lambda_conf * loss_conf
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
    
    model.load_state_dict(best_state)
    return model


# Test
if __name__ == "__main__":
    from data_loader import load_dataset
    from models import TabularMLP
    
    X_lab, y_lab, X_unlab, X_test, y_test = load_dataset('adult', label_fraction=0.1)
    
    # Split labeled data into train/val
    val_size = int(0.2 * len(X_lab))
    X_train, X_val = X_lab[:-val_size], X_lab[-val_size:]
    y_train, y_val = y_lab[:-val_size], y_lab[-val_size:]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test supervised training
    print("Testing supervised training...")
    model = TabularMLP(input_dim=X_train.shape[1], num_classes=2)
    model = train_supervised(model, X_train, y_train, X_val, y_val, 
                            epochs=50, device=device)
    print("✅ Supervised training works!")
    
    # Test calibration-aware training
    print("\nTesting calibration-aware training...")
    model2 = TabularMLP(input_dim=X_train.shape[1], num_classes=2)
    model2 = train_with_calibration_reg(model2, X_train, y_train, X_val, y_val,
                                        epochs=50, lambda_conf=0.1, device=device)
    print("✅ Calibration-aware training works!")