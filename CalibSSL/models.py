import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import numpy as np

class TreeBaseline:
    """Wrapper for tree-based models"""
    
    def __init__(self, model_type='xgboost', n_classes=2, calibrate=False):
        self.model_type = model_type
        self.n_classes = n_classes
        self.calibrate = calibrate
        
        if model_type == 'random_forest':
            self.base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'xgboost':
            if n_classes == 2:
                objective = 'binary:logistic'
            else:
                objective = 'multi:softprob'
            
            self.base_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective=objective,
                num_class=n_classes if n_classes > 2 else None,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Optionally wrap with calibration
        if calibrate:
            self.model = CalibratedClassifierCV(
                self.base_model, 
                method='isotonic',  # or 'sigmoid'
                cv=3
            )
        else:
            self.model = self.base_model
    
    def fit(self, X, y):
        """Train the model"""
        # Convert torch tensors to numpy if needed
        if hasattr(X, 'numpy'):
            X = X.cpu().numpy()
        if hasattr(y, 'numpy'):
            y = y.cpu().numpy()
        
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Get probability predictions"""
        if hasattr(X, 'numpy'):
            X = X.cpu().numpy()
        
        probs = self.model.predict_proba(X)
        return probs
    
    def predict(self, X):
        """Get class predictions"""
        if hasattr(X, 'numpy'):
            X = X.cpu().numpy()
        
        return self.model.predict(X)
    
    def score(self, X, y):
        """Get accuracy"""
        if hasattr(X, 'numpy'):
            X = X.cpu().numpy()
        if hasattr(y, 'numpy'):
            y = y.cpu().numpy()
        
        return self.model.score(X, y)


# Test tree models
if __name__ == "__main__":
    from data_loader import load_dataset
    
    print("Testing tree-based models...")
    X_lab, y_lab, X_unlab, X_test, y_test = load_dataset('adult', label_fraction=0.1)
    
    # Test Random Forest
    print("\n1. Random Forest:")
    rf = TreeBaseline(model_type='random_forest', n_classes=2, calibrate=False)
    rf.fit(X_lab, y_lab)
    acc = rf.score(X_test, y_test)
    probs = rf.predict_proba(X_test)
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Probs shape: {probs.shape}")
    
    # Test XGBoost
    print("\n2. XGBoost:")
    xgb_model = TreeBaseline(model_type='xgboost', n_classes=2, calibrate=False)
    xgb_model.fit(X_lab, y_lab)
    acc = xgb_model.score(X_test, y_test)
    probs = xgb_model.predict_proba(X_test)
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Probs shape: {probs.shape}")
    
    # Test XGBoost with calibration
    print("\n3. XGBoost (Calibrated):")
    xgb_calib = TreeBaseline(model_type='xgboost', n_classes=2, calibrate=True)
    xgb_calib.fit(X_lab, y_lab)
    acc = xgb_calib.score(X_test, y_test)
    probs = xgb_calib.predict_proba(X_test)
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Probs shape: {probs.shape}")
    
    print("\n✅ All tree models working!")
class TabularMLP(nn.Module):
    """Standard MLP for tabular data"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], num_classes=2, dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_representations(self, x):
        """Get penultimate layer representations"""
        for layer in self.network[:-1]:
            x = layer(x)
        return x


class ViMEEncoder(nn.Module):
    """ViME: Value Imputation and Mask Estimation"""
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Decoder (for reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Mask predictor
        self.mask_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x, mask):
        # Corrupt input
        x_corrupted = x * (1 - mask)
        
        # Encode
        z = self.encoder(x_corrupted)
        
        # Reconstruct
        x_recon = self.decoder(z)
        
        # Predict mask
        mask_pred = self.mask_predictor(z)
        
        return x_recon, mask_pred, z


# Test models
if __name__ == "__main__":
    # Test MLP
    model = TabularMLP(input_dim=50, hidden_dims=[128, 64], num_classes=2)
    x = torch.randn(32, 50)
    out = model(x)
    print(f"MLP output shape: {out.shape}")  # Should be [32, 2]
    
    # Test ViME
    vime = ViMEEncoder(input_dim=50, hidden_dim=128)
    mask = torch.bernoulli(torch.full((32, 50), 0.3))
    x_recon, mask_pred, z = vime(x, mask)
    print(f"ViME reconstruction shape: {x_recon.shape}")  # [32, 50]
    print(f"ViME mask prediction shape: {mask_pred.shape}")  # [32, 50]
    print(f"ViME representation shape: {z.shape}")  # [32, 128]