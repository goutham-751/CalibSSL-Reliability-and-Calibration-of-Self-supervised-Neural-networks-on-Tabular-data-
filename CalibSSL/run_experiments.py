"""
Main experimental script for CalibSSL paper

Runs all experiments:
- 5 datasets (adult, bank, credit, covertype, diabetes)
- 5 label fractions (5%, 10%, 15%, 20%, 100%)
- 7 models (RF, XGB, XGB-Calib, MLP, SSL-MLP, CalibSSL, MLP-Calib)
= 175 total experiments

Estimated time: 10-15 hours on GPU
"""

import pandas as pd
import torch
import os
from datetime import datetime
from tqdm import tqdm

from data_loader import load_dataset
from models import TabularMLP, TreeBaseline, ViMEEncoder
from ssl_pretrain import pretrain_vime
from train import train_supervised, train_with_calibration_reg
from evaluate import evaluate_model

def run_all_experiments(quick_test=False):
    """
    Run complete experimental suite
    
    Args:
        quick_test: If True, run reduced experiments for testing
    """
    
    # Configuration
    if quick_test:
        print("⚠️  QUICK TEST MODE - Running reduced experiments")
        datasets = ['adult', 'diabetes']  # Just 2 datasets
        label_fractions = [0.05, 0.20]    # Just 2 fractions
    else:
        datasets = ['adult', 'bank', 'credit', 'covertype', 'diabetes']
        label_fractions = [0.05, 0.10, 0.15, 0.20, 1.0]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    
    print("="*70)
    print("CALIBSSL EXPERIMENTAL SUITE")
    print("="*70)
    print(f"Device: {device}")
    print(f"Datasets: {len(datasets)}")
    print(f"Label fractions: {len(label_fractions)}")
    print(f"Models: 7")
    print(f"Total experiments: {len(datasets) * len(label_fractions) * 7}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    results = []
    experiment_id = 0
    
    # Main experimental loop
    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}\n")
        
        for label_frac in label_fractions:
            print(f"\n{'─'*70}")
            print(f"Label Fraction: {label_frac*100:.0f}%")
            print(f"{'─'*70}")
            
            # ========== LOAD DATA ==========
            print("\n[1/8] Loading data...")
            X_lab, y_lab, X_unlab, X_test, y_test = load_dataset(
                dataset_name, 
                label_fraction=label_frac
            )
            
            # Split labeled data into train/val (80/20)
            val_size = int(0.2 * len(X_lab))
            X_train = X_lab[:-val_size]
            y_train = y_lab[:-val_size]
            X_val = X_lab[-val_size:]
            y_val = y_lab[-val_size:]
            
            input_dim = X_train.shape[1]
            n_classes = len(torch.unique(y_test))
            
            print(f"      Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
            print(f"      Features: {input_dim} | Classes: {n_classes}")
            
            # ========== MODEL 1: RANDOM FOREST ==========
            print("\n[2/8] Training Random Forest...")
            rf = TreeBaseline('random_forest', n_classes=n_classes, calibrate=False)
            rf.fit(X_train, y_train)
            metrics_rf = evaluate_model(rf, X_test, y_test, model_type='tree')
            
            results.append({
                'experiment_id': experiment_id,
                'dataset': dataset_name,
                'label_fraction': label_frac,
                'model': 'Random_Forest',
                **metrics_rf
            })
            experiment_id += 1
            print(f"      ✓ Acc: {metrics_rf['accuracy']:.4f} | ECE: {metrics_rf['ece']:.4f} | Brier: {metrics_rf['brier']:.4f}")
            
            # ========== MODEL 2: XGBOOST ==========
            print("\n[3/8] Training XGBoost...")
            xgb = TreeBaseline('xgboost', n_classes=n_classes, calibrate=False)
            xgb.fit(X_train, y_train)
            metrics_xgb = evaluate_model(xgb, X_test, y_test, model_type='tree')
            
            results.append({
                'experiment_id': experiment_id,
                'dataset': dataset_name,
                'label_fraction': label_frac,
                'model': 'XGBoost',
                **metrics_xgb
            })
            experiment_id += 1
            print(f"      ✓ Acc: {metrics_xgb['accuracy']:.4f} | ECE: {metrics_xgb['ece']:.4f} | Brier: {metrics_xgb['brier']:.4f}")
            
            # ========== MODEL 3: XGBOOST (CALIBRATED) ==========
            print("\n[4/8] Training XGBoost (Calibrated)...")
            xgb_calib = TreeBaseline('xgboost', n_classes=n_classes, calibrate=True)
            xgb_calib.fit(X_train, y_train)
            metrics_xgb_calib = evaluate_model(xgb_calib, X_test, y_test, model_type='tree')
            
            results.append({
                'experiment_id': experiment_id,
                'dataset': dataset_name,
                'label_fraction': label_frac,
                'model': 'XGBoost_Calibrated',
                **metrics_xgb_calib
            })
            experiment_id += 1
            print(f"      ✓ Acc: {metrics_xgb_calib['accuracy']:.4f} | ECE: {metrics_xgb_calib['ece']:.4f} | Brier: {metrics_xgb_calib['brier']:.4f}")
            
            # ========== MODEL 4: SUPERVISED MLP (BASELINE) ==========
            print("\n[5/8] Training Supervised MLP (baseline)...")
            mlp_sup = TabularMLP(input_dim=input_dim, num_classes=n_classes)
            mlp_sup = train_supervised(
                mlp_sup, X_train, y_train, X_val, y_val,
                epochs=100, device=device, patience=15, verbose=False
            )
            metrics_mlp = evaluate_model(mlp_sup, X_test, y_test, model_type='neural')
            
            results.append({
                'experiment_id': experiment_id,
                'dataset': dataset_name,
                'label_fraction': label_frac,
                'model': 'Supervised_MLP',
                **metrics_mlp
            })
            experiment_id += 1
            print(f"      ✓ Acc: {metrics_mlp['accuracy']:.4f} | ECE: {metrics_mlp['ece']:.4f} | Brier: {metrics_mlp['brier']:.4f}")
            
            # ========== SSL PRETRAINING (for models 5-7) ==========
            print("\n[6/8] SSL Pretraining (ViME)...")
            pretrained_encoder = pretrain_vime(
                X_unlab,
                input_dim=input_dim,
                hidden_dim=256,
                epochs=50,
                mask_prob=0.3,
                device=device
            )
            print("      ✓ Pretraining complete")
            
            # ========== MODEL 5: SSL-MLP ==========
            print("\n[7/8] Training SSL-MLP...")
            mlp_ssl = TabularMLP(input_dim=input_dim, num_classes=n_classes)
            
            # Use pretrained encoder weights
            with torch.no_grad():
                mlp_ssl.network[0].weight.data = pretrained_encoder[0].weight.data.clone()
                mlp_ssl.network[0].bias.data = pretrained_encoder[0].bias.data.clone()
            
            mlp_ssl = train_supervised(
                mlp_ssl, X_train, y_train, X_val, y_val,
                epochs=100, device=device, patience=15, verbose=False
            )
            metrics_ssl = evaluate_model(mlp_ssl, X_test, y_test, model_type='neural')
            
            results.append({
                'experiment_id': experiment_id,
                'dataset': dataset_name,
                'label_fraction': label_frac,
                'model': 'SSL_MLP',
                **metrics_ssl
            })
            experiment_id += 1
            print(f"      ✓ Acc: {metrics_ssl['accuracy']:.4f} | ECE: {metrics_ssl['ece']:.4f} | Brier: {metrics_ssl['brier']:.4f}")
            
            # ========== MODEL 6: CalibSSL (OUR METHOD) ==========
            print("\n[8/8] Training CalibSSL (SSL + Calibration)...")
            mlp_calibssl = TabularMLP(input_dim=input_dim, num_classes=n_classes)
            
            # Use pretrained encoder
            with torch.no_grad():
                mlp_calibssl.network[0].weight.data = pretrained_encoder[0].weight.data.clone()
                mlp_calibssl.network[0].bias.data = pretrained_encoder[0].bias.data.clone()
            
            mlp_calibssl = train_with_calibration_reg(
                mlp_calibssl, X_train, y_train, X_val, y_val,
                epochs=100, lambda_conf=0.1, device=device, patience=15, verbose=False
            )
            metrics_calibssl = evaluate_model(mlp_calibssl, X_test, y_test, model_type='neural')
            
            results.append({
                'experiment_id': experiment_id,
                'dataset': dataset_name,
                'label_fraction': label_frac,
                'model': 'CalibSSL',
                **metrics_calibssl
            })
            experiment_id += 1
            print(f"      ✓ Acc: {metrics_calibssl['accuracy']:.4f} | ECE: {metrics_calibssl['ece']:.4f} | Brier: {metrics_calibssl['brier']:.4f}")
            
            # ========== MODEL 7: MLP + Calibration Only (Ablation) ==========
            print("\n[Ablation] Training MLP + Calibration (no SSL)...")
            mlp_calib_only = TabularMLP(input_dim=input_dim, num_classes=n_classes)
            mlp_calib_only = train_with_calibration_reg(
                mlp_calib_only, X_train, y_train, X_val, y_val,
                epochs=100, lambda_conf=0.1, device=device, patience=15, verbose=False
            )
            metrics_calib_only = evaluate_model(mlp_calib_only, X_test, y_test, model_type='neural')
            
            results.append({
                'experiment_id': experiment_id,
                'dataset': dataset_name,
                'label_fraction': label_frac,
                'model': 'MLP_Calib_Only',
                **metrics_calib_only
            })
            experiment_id += 1
            print(f"      ✓ Acc: {metrics_calib_only['accuracy']:.4f} | ECE: {metrics_calib_only['ece']:.4f} | Brier: {metrics_calib_only['brier']:.4f}")
            
            # Save intermediate results
            df_results = pd.DataFrame(results)
            df_results.to_csv('results/results.csv', index=False)
            
            print(f"\n      💾 Progress saved: {experiment_id} experiments completed")
    
    # ========== SAVE FINAL RESULTS ==========
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/results.csv', index=False)
    
    print("\n" + "="*70)
    print("✅ ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"Total experiments: {len(results)}")
    print(f"Results saved to: results/results.csv")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return df_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run CalibSSL experiments')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run quick test with reduced experiments')
    args = parser.parse_args()
    
    # Run experiments
    results_df = run_all_experiments(quick_test=args.quick_test)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY BY MODEL (averaged across all experiments):")
    print("="*70)
    summary = results_df.groupby('model')[['accuracy', 'f1', 'ece', 'brier']].mean()
    print(summary.round(4).to_string())
    print("\n")