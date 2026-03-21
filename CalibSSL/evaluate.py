import torch
import numpy as np
from netcal.metrics import ECE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

def evaluate_model(model, X_test, y_test, model_type='neural'):
    """
    Comprehensive evaluation for both neural and tree models
    
    Args:
        model: Trained model (neural or tree)
        X_test: Test features
        y_test: Test labels
        model_type: 'neural' or 'tree'
    
    Returns:
        Dictionary of metrics
    """
    
    if model_type == 'neural':
        # Neural network evaluation
        model.eval()
        device = next(model.parameters()).device
        X_test = X_test.to(device)
        y_test_np = y_test.cpu().numpy() if hasattr(y_test, 'cpu') else y_test
        
        with torch.no_grad():
            logits = model(X_test)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
    
    else:  # tree
        # Tree model evaluation
        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)
        y_test_np = y_test.cpu().numpy() if hasattr(y_test, 'cpu') else y_test
    
    # Number of classes
    n_classes = probs.shape[1]
    
    # ========== PERFORMANCE METRICS ==========
    accuracy = accuracy_score(y_test_np, preds)
    
    # F1 score (macro for multi-class, binary for binary)
    f1_avg = 'binary' if n_classes == 2 else 'macro'
    f1 = f1_score(y_test_np, preds, average=f1_avg, zero_division=0)
    
    # AUC (only for binary classification)
    if n_classes == 2:
        auc = roc_auc_score(y_test_np, probs[:, 1])
    else:
        # Multi-class AUC (one-vs-rest)
        try:
            auc = roc_auc_score(y_test_np, probs, multi_class='ovr', average='macro')
        except:
            auc = np.nan
    
    # ========== CALIBRATION METRICS ==========
    
    # Expected Calibration Error (ECE)
    ece_calculator = ECE(bins=15)
    ece = ece_calculator.measure(probs, y_test_np)
    
    # Maximum Calibration Error (MCE)
    from netcal.metrics import MCE
    mce_calculator = MCE(bins=15)
    mce = mce_calculator.measure(probs, y_test_np)
    
    # Brier Score
    # For multi-class, need one-hot encoding
    if n_classes == 2:
        brier = brier_score_loss(y_test_np, probs[:, 1])
    else:
        # One-hot encode targets
        y_one_hot = np.zeros((len(y_test_np), n_classes))
        y_one_hot[np.arange(len(y_test_np)), y_test_np] = 1
        brier = np.mean((probs - y_one_hot) ** 2)
    
    # Confidence (average max probability)
    confidence = probs.max(axis=1).mean()
    
    # Return all metrics
    return {
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'ece': ece,
        'mce': mce,
        'brier': brier,
        'confidence': confidence
    }


def get_reliability_diagram_data(probs, y_true, n_bins=10):
    """
    Get data for plotting reliability diagrams
    
    Returns:
        bin_centers, bin_accuracies, bin_confidences, bin_counts
    """
    # Get predicted class and confidence
    pred_probs = probs.max(axis=1)
    pred_class = probs.argmax(axis=1)
    
    # Check correctness
    correct = (pred_class == y_true).astype(float)
    
    # Bin predictions by confidence
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(pred_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_accuracies.append(correct[mask].mean())
            bin_confidences.append(pred_probs[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)
            bin_counts.append(0)
    
    return np.array(bin_centers), np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_counts)


# Test evaluation
if __name__ == "__main__":
    from data_loader import load_dataset
    from models import TabularMLP, TreeBaseline
    from train import train_supervised
    
    print("Testing evaluation pipeline...")
    X_lab, y_lab, X_unlab, X_test, y_test = load_dataset('adult', label_fraction=0.1)
    
    val_size = int(0.2 * len(X_lab))
    X_train, X_val = X_lab[:-val_size], X_lab[-val_size:]
    y_train, y_val = y_lab[:-val_size], y_lab[-val_size:]
    
    # Test with neural network
    print("\n1. Evaluating Neural Network:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_nn = TabularMLP(input_dim=X_train.shape[1], num_classes=2)
    model_nn = train_supervised(model_nn, X_train, y_train, X_val, y_val, 
                               epochs=20, device=device)
    
    metrics_nn = evaluate_model(model_nn, X_test, y_test, model_type='neural')
    print(f"   Accuracy: {metrics_nn['accuracy']:.4f}")
    print(f"   F1: {metrics_nn['f1']:.4f}")
    print(f"   ECE: {metrics_nn['ece']:.4f}")
    print(f"   Brier: {metrics_nn['brier']:.4f}")
    
    # Test with XGBoost
    print("\n2. Evaluating XGBoost:")
    model_xgb = TreeBaseline(model_type='xgboost', n_classes=2, calibrate=False)
    model_xgb.fit(X_train, y_train)
    
    metrics_xgb = evaluate_model(model_xgb, X_test, y_test, model_type='tree')
    print(f"   Accuracy: {metrics_xgb['accuracy']:.4f}")
    print(f"   F1: {metrics_xgb['f1']:.4f}")
    print(f"   ECE: {metrics_xgb['ece']:.4f}")
    print(f"   Brier: {metrics_xgb['brier']:.4f}")
    
    print("\n✅ Evaluation pipeline working!")