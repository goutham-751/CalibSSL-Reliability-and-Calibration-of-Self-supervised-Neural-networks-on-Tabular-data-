"""
Enhanced CalibSSL Experimental Suite
- Multi-seed runs (3 seeds) for statistical robustness
- Temperature Scaling as additional baseline
- Reliability diagram data collection
- Lambda_conf fixed at 0.1

Estimated time: ~8-10 hours on CPU
"""

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from data_loader import load_dataset
from models import TabularMLP, TreeBaseline, ViMEEncoder
from ssl_pretrain import pretrain_vime
from train import train_supervised, train_with_calibration_reg
from evaluate import evaluate_model, get_reliability_diagram_data


# ============================================================
# TEMPERATURE SCALING
# ============================================================

class TemperatureScaling:
    """Post-hoc temperature scaling for neural network calibration"""
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, model, X_val, y_val, device='cpu'):
        """Learn optimal temperature on validation set"""
        model.eval()
        X_val_d = X_val.to(device)
        y_val_d = y_val.to(device)
        
        with torch.no_grad():
            logits = model(X_val_d)
        
        temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()
        
        def eval_fn():
            optimizer.zero_grad()
            loss = criterion(logits / temperature, y_val_d)
            loss.backward()
            return loss
        
        optimizer.step(eval_fn)
        self.temperature = max(temperature.item(), 0.1)  # Prevent extreme values
        return self
    
    def evaluate(self, model, X_test, y_test, device='cpu'):
        """Get temperature-scaled predictions and evaluate"""
        from netcal.metrics import ECE, MCE
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
        
        model.eval()
        X_test_d = X_test.to(device)
        
        with torch.no_grad():
            logits = model(X_test_d)
            probs = torch.softmax(logits / self.temperature, dim=1).cpu().numpy()
        
        preds = probs.argmax(axis=1)
        y_np = y_test.cpu().numpy() if hasattr(y_test, 'cpu') else y_test
        n_classes = probs.shape[1]
        
        accuracy = accuracy_score(y_np, preds)
        f1_avg = 'binary' if n_classes == 2 else 'macro'
        f1 = f1_score(y_np, preds, average=f1_avg, zero_division=0)
        
        if n_classes == 2:
            auc = roc_auc_score(y_np, probs[:, 1])
        else:
            try:
                auc = roc_auc_score(y_np, probs, multi_class='ovr', average='macro')
            except:
                auc = np.nan
        
        ece = ECE(bins=15).measure(probs, y_np)
        mce = MCE(bins=15).measure(probs, y_np)
        
        if n_classes == 2:
            brier = brier_score_loss(y_np, probs[:, 1])
        else:
            y_oh = np.zeros((len(y_np), n_classes))
            y_oh[np.arange(len(y_np)), y_np] = 1
            brier = np.mean((probs - y_oh) ** 2)
        
        confidence = probs.max(axis=1).mean()
        
        return {
            'accuracy': accuracy, 'f1': f1, 'auc': auc,
            'ece': ece, 'mce': mce, 'brier': brier,
            'confidence': confidence
        }, probs


# ============================================================
# RELIABILITY DIAGRAMS
# ============================================================

def plot_reliability_diagrams(reliability_data, save_dir='figures'):
    """Generate publication-quality reliability diagrams"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Group by dataset
    datasets = sorted(set(d['dataset'] for d in reliability_data))
    models_to_plot = ['Supervised_MLP', 'SSL_MLP', 'CalibSSL', 'CalibSSL_TempScaled']
    
    colors = {
        'Supervised_MLP': '#999999',
        'SSL_MLP': '#56B4E9',
        'CalibSSL': '#E69F00',
        'CalibSSL_TempScaled': '#D55E00'
    }
    labels = {
        'Supervised_MLP': 'Supervised',
        'SSL_MLP': 'SSL-MLP',
        'CalibSSL': 'CalibSSL (Ours)',
        'CalibSSL_TempScaled': 'CalibSSL + TempScale'
    }
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Perfect')
        
        for model in models_to_plot:
            entries = [d for d in reliability_data 
                      if d['dataset'] == dataset and d['model'] == model 
                      and d['label_fraction'] == 0.1]
            
            if entries:
                entry = entries[0]
                centers = entry['bin_centers']
                accs = entry['bin_accuracies']
                
                # Filter NaN bins
                mask = ~np.isnan(accs)
                if mask.any():
                    ax.plot(centers[mask], accs[mask], 'o-',
                           color=colors.get(model, 'gray'),
                           label=labels.get(model, model),
                           linewidth=2, markersize=6)
        
        ax.set_xlabel('Confidence', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title(f'{dataset.capitalize()}', fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.suptitle('Reliability Diagrams (10% Labels)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fig8_reliability_diagrams.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/fig8_reliability_diagrams.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Figure 8: Reliability Diagrams")


# ============================================================
# MAIN ENHANCED EXPERIMENT RUNNER
# ============================================================

def run_enhanced_experiments():
    """
    Run complete experiment suite with:
    - 3 random seeds for robustness
    - 8 models (original 7 + Temperature Scaling)
    - Reliability diagram data collection
    """
    
    # Using 'jannis' (83k rows) instead of 'covertype' (581k rows) for local tracking
    datasets = ['adult', 'bank', 'credit', 'jannis', 'diabetes']
    label_fractions = [0.05, 0.10, 0.15, 0.20, 1.0]
    seeds = [42, 123, 456]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    
    total_exps = len(datasets) * len(label_fractions) * 8 * len(seeds)
    
    print("=" * 70)
    print("ENHANCED CALIBSSL EXPERIMENTAL SUITE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Datasets: {len(datasets)}")
    print(f"Label fractions: {len(label_fractions)}")
    print(f"Models: 8 (original 7 + Temp Scaling)")
    print(f"Seeds: {seeds}")
    print(f"Total experiments: {total_exps}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    
    all_results = []
    reliability_data = []
    experiment_id = 0
    completed_combos = set()
    
    if os.path.exists('results/results.csv'):
        try:
            df_existing = pd.read_csv('results/results.csv')
            if len(df_existing) > 0:
                # Count models per combo
                combo_counts = df_existing.groupby(['dataset', 'label_fraction', 'seed']).size()
                completed_combos = set(combo_counts[combo_counts == 8].index)
                
                # Keep only fully completed combos to avoid duplicates from interrupted runs
                keys = list(zip(df_existing.dataset, df_existing.label_fraction, df_existing.seed))
                df_clean = df_existing[[k in completed_combos for k in keys]]
                
                all_results = df_clean.to_dict('records')
                experiment_id = len(all_results)
                print(f"  ⏩ Resumed {len(completed_combos)} fully completed combos ({experiment_id} saved models)")
        except Exception as e:
            print(f"Warning: Could not load existing results.csv: {e}")
            
    if os.path.exists('results/reliability_data.npy'):
        try:
            reliability_data = np.load('results/reliability_data.npy', allow_pickle=True).tolist()
            print(f"  ⏩ Resumed reliability diagram data")
        except:
            pass
    
    for seed in seeds:
        print(f"\n{'*' * 70}")
        print(f"  SEED = {seed}")
        print(f"{'*' * 70}")
        
        # Set all random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        for dataset_name in datasets:
            print(f"\n{'=' * 70}")
            print(f"DATASET: {dataset_name.upper()} | Seed: {seed}")
            print(f"{'=' * 70}\n")
            
            for label_frac in label_fractions:
                print(f"\n{'─' * 70}")
                print(f"Label Fraction: {label_frac*100:.0f}% | Seed: {seed}")
                print(f"{'─' * 70}")
                
                if (dataset_name, label_frac, seed) in completed_combos:
                    print("  ⏩ Already completed. Skipping.")
                    continue
                
                # Load data with seed
                X_lab, y_lab, X_unlab, X_test, y_test = load_dataset(
                    dataset_name, label_fraction=label_frac
                )
                
                val_size = int(0.2 * len(X_lab))
                X_train = X_lab[:-val_size]
                y_train = y_lab[:-val_size]
                X_val = X_lab[-val_size:]
                y_val = y_lab[-val_size:]
                
                input_dim = X_train.shape[1]
                n_classes = len(torch.unique(y_test))
                
                print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
                
                base_info = {
                    'dataset': dataset_name,
                    'label_fraction': label_frac,
                    'seed': seed
                }
                
                # ========== MODEL 1: RANDOM FOREST ==========
                print("  [1/9] Random Forest...", end=" ")
                rf = TreeBaseline('random_forest', n_classes=n_classes, calibrate=False)
                rf.fit(X_train, y_train)
                m = evaluate_model(rf, X_test, y_test, model_type='tree')
                all_results.append({**base_info, 'experiment_id': experiment_id, 'model': 'Random_Forest', **m})
                experiment_id += 1
                print(f"Acc:{m['accuracy']:.4f} ECE:{m['ece']:.4f}")
                
                # ========== MODEL 2: XGBOOST ==========
                print("  [2/9] XGBoost...", end=" ")
                xgb = TreeBaseline('xgboost', n_classes=n_classes, calibrate=False)
                xgb.fit(X_train, y_train)
                m = evaluate_model(xgb, X_test, y_test, model_type='tree')
                all_results.append({**base_info, 'experiment_id': experiment_id, 'model': 'XGBoost', **m})
                experiment_id += 1
                print(f"Acc:{m['accuracy']:.4f} ECE:{m['ece']:.4f}")
                
                # ========== MODEL 3: XGBOOST CALIBRATED ==========
                print("  [3/9] XGBoost (Calibrated)...", end=" ")
                xgb_c = TreeBaseline('xgboost', n_classes=n_classes, calibrate=True)
                xgb_c.fit(X_train, y_train)
                m = evaluate_model(xgb_c, X_test, y_test, model_type='tree')
                all_results.append({**base_info, 'experiment_id': experiment_id, 'model': 'XGBoost_Calibrated', **m})
                experiment_id += 1
                print(f"Acc:{m['accuracy']:.4f} ECE:{m['ece']:.4f}")
                
                # ========== MODEL 4: SUPERVISED MLP ==========
                print("  [4/9] Supervised MLP...", end=" ")
                mlp_sup = TabularMLP(input_dim=input_dim, num_classes=n_classes)
                mlp_sup = train_supervised(
                    mlp_sup, X_train, y_train, X_val, y_val,
                    epochs=100, device=device, patience=15, verbose=False
                )
                m = evaluate_model(mlp_sup, X_test, y_test, model_type='neural')
                all_results.append({**base_info, 'experiment_id': experiment_id, 'model': 'Supervised_MLP', **m})
                experiment_id += 1
                print(f"Acc:{m['accuracy']:.4f} ECE:{m['ece']:.4f}")
                
                # Collect reliability data for supervised
                if seed == seeds[0]:
                    mlp_sup.eval()
                    with torch.no_grad():
                        probs_sup = torch.softmax(mlp_sup(X_test.to(device)), dim=1).cpu().numpy()
                    y_np = y_test.cpu().numpy()
                    bc, ba, bconf, bcount = get_reliability_diagram_data(probs_sup, y_np)
                    reliability_data.append({
                        'dataset': dataset_name, 'label_fraction': label_frac,
                        'model': 'Supervised_MLP', 'bin_centers': bc, 'bin_accuracies': ba
                    })
                
                # ========== SSL PRETRAINING ==========
                print("  [5/9] SSL Pretraining...", end=" ")
                pretrained_encoder = pretrain_vime(
                    X_unlab, input_dim=input_dim, hidden_dim=256,
                    epochs=50, mask_prob=0.3, device=device
                )
                print("done")
                
                # ========== MODEL 5: SSL-MLP ==========
                print("  [6/9] SSL-MLP...", end=" ")
                mlp_ssl = TabularMLP(input_dim=input_dim, num_classes=n_classes)
                with torch.no_grad():
                    mlp_ssl.network[0].weight.data = pretrained_encoder[0].weight.data.clone()
                    mlp_ssl.network[0].bias.data = pretrained_encoder[0].bias.data.clone()
                
                mlp_ssl = train_supervised(
                    mlp_ssl, X_train, y_train, X_val, y_val,
                    epochs=100, device=device, patience=15, verbose=False
                )
                m = evaluate_model(mlp_ssl, X_test, y_test, model_type='neural')
                all_results.append({**base_info, 'experiment_id': experiment_id, 'model': 'SSL_MLP', **m})
                experiment_id += 1
                print(f"Acc:{m['accuracy']:.4f} ECE:{m['ece']:.4f}")
                
                # Collect reliability data
                if seed == seeds[0]:
                    mlp_ssl.eval()
                    with torch.no_grad():
                        probs_ssl = torch.softmax(mlp_ssl(X_test.to(device)), dim=1).cpu().numpy()
                    bc, ba, _, _ = get_reliability_diagram_data(probs_ssl, y_np)
                    reliability_data.append({
                        'dataset': dataset_name, 'label_fraction': label_frac,
                        'model': 'SSL_MLP', 'bin_centers': bc, 'bin_accuracies': ba
                    })
                
                # ========== MODEL 6: CalibSSL ==========
                print("  [7/9] CalibSSL...", end=" ")
                mlp_calibssl = TabularMLP(input_dim=input_dim, num_classes=n_classes)
                with torch.no_grad():
                    mlp_calibssl.network[0].weight.data = pretrained_encoder[0].weight.data.clone()
                    mlp_calibssl.network[0].bias.data = pretrained_encoder[0].bias.data.clone()
                
                mlp_calibssl = train_with_calibration_reg(
                    mlp_calibssl, X_train, y_train, X_val, y_val,
                    epochs=100, lambda_conf=0.1, device=device, patience=15, verbose=False
                )
                m = evaluate_model(mlp_calibssl, X_test, y_test, model_type='neural')
                all_results.append({**base_info, 'experiment_id': experiment_id, 'model': 'CalibSSL', **m})
                experiment_id += 1
                print(f"Acc:{m['accuracy']:.4f} ECE:{m['ece']:.4f}")
                
                # Collect reliability data
                if seed == seeds[0]:
                    mlp_calibssl.eval()
                    with torch.no_grad():
                        probs_cal = torch.softmax(mlp_calibssl(X_test.to(device)), dim=1).cpu().numpy()
                    bc, ba, _, _ = get_reliability_diagram_data(probs_cal, y_np)
                    reliability_data.append({
                        'dataset': dataset_name, 'label_fraction': label_frac,
                        'model': 'CalibSSL', 'bin_centers': bc, 'bin_accuracies': ba
                    })
                
                # ========== MODEL 7: CalibSSL + Temperature Scaling ==========
                print("  [8/9] CalibSSL + TempScaling...", end=" ")
                ts = TemperatureScaling()
                ts.fit(mlp_calibssl, X_val, y_val, device=device)
                m_ts, probs_ts = ts.evaluate(mlp_calibssl, X_test, y_test, device=device)
                all_results.append({**base_info, 'experiment_id': experiment_id, 'model': 'CalibSSL_TempScaled', **m_ts})
                experiment_id += 1
                print(f"Acc:{m_ts['accuracy']:.4f} ECE:{m_ts['ece']:.4f} T={ts.temperature:.2f}")
                
                # Collect reliability data
                if seed == seeds[0]:
                    bc, ba, _, _ = get_reliability_diagram_data(probs_ts, y_np)
                    reliability_data.append({
                        'dataset': dataset_name, 'label_fraction': label_frac,
                        'model': 'CalibSSL_TempScaled', 'bin_centers': bc, 'bin_accuracies': ba
                    })
                
                # ========== MODEL 8: MLP + Calibration Only ==========
                print("  [9/9] MLP + Calib Only...", end=" ")
                mlp_co = TabularMLP(input_dim=input_dim, num_classes=n_classes)
                mlp_co = train_with_calibration_reg(
                    mlp_co, X_train, y_train, X_val, y_val,
                    epochs=100, lambda_conf=0.1, device=device, patience=15, verbose=False
                )
                m = evaluate_model(mlp_co, X_test, y_test, model_type='neural')
                all_results.append({**base_info, 'experiment_id': experiment_id, 'model': 'MLP_Calib_Only', **m})
                experiment_id += 1
                print(f"Acc:{m['accuracy']:.4f} ECE:{m['ece']:.4f}")
                
                # Save intermediate results
                df_results = pd.DataFrame(all_results)
                df_results.to_csv('results/results.csv', index=False)
                np.save('results/reliability_data.npy', reliability_data)
                print(f"  💾 Progress saved: {experiment_id} experiments completed")
    
    # ========== FINAL SUMMARY ==========
    df_results = pd.DataFrame(all_results)
    df_results.to_csv('results/results.csv', index=False)
    
    print("\n" + "=" * 70)
    print("✅ ALL EXPERIMENTS COMPLETED!")
    print("=" * 70)
    print(f"Total experiments: {len(all_results)}")
    print(f"Results saved to: results/results.csv")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary (averaged across seeds)
    print("\n" + "=" * 70)
    print("SUMMARY BY MODEL (mean ± std across all experiments and seeds):")
    print("=" * 70)
    
    summary = df_results.groupby('model').agg(
        acc_mean=('accuracy', 'mean'),
        acc_std=('accuracy', 'std'),
        ece_mean=('ece', 'mean'),
        ece_std=('ece', 'std'),
        brier_mean=('brier', 'mean'),
        brier_std=('brier', 'std')
    ).round(4)
    
    print(f"\n{'Model':<25s} {'Accuracy':>18s} {'ECE':>18s} {'Brier':>18s}")
    print("-" * 85)
    for model, row in summary.iterrows():
        print(f"{model:<25s} {row['acc_mean']:.4f}±{row['acc_std']:.4f}  "
              f"{row['ece_mean']:.4f}±{row['ece_std']:.4f}  "
              f"{row['brier_mean']:.4f}±{row['brier_std']:.4f}")
    
    # Save summary
    summary.to_csv('results/summary_with_seeds.csv')
    print("\n✓ Saved: results/summary_with_seeds.csv")
    
    # Generate reliability diagrams
    print("\nGenerating reliability diagrams...")
    plot_reliability_diagrams(reliability_data)
    
    print("\n" + "=" * 70)
    print(" DONE! Next steps:")
    print("  python analyze_results.py   # Statistical analysis")
    print("  python visualize.py         # All publication figures")
    print("  python statistical_tests.py # Significance tests")
    print("  python error_analysis.py    # Failure analysis")
    print("=" * 70)
    
    return df_results


if __name__ == "__main__":
    results_df = run_enhanced_experiments()
