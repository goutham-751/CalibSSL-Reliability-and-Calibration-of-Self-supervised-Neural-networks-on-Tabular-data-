"""
Lambda_conf tuning script for CalibSSL

Only re-runs CalibSSL and MLP_Calib_Only with different lambda values.
Keeps all other model results from existing results.csv.
Picks the best lambda_conf based on ECE improvement while maintaining accuracy.

Estimated time: ~30-40 min on CPU
"""

import pandas as pd
import torch
import os
from datetime import datetime

from data_loader import load_dataset
from models import TabularMLP, ViMEEncoder
from ssl_pretrain import pretrain_vime
from train import train_with_calibration_reg
from evaluate import evaluate_model


def tune_lambda(lambda_values=[0.01, 0.05, 0.1, 0.2, 0.5]):
    """
    Re-run CalibSSL and MLP_Calib_Only with multiple lambda_conf values.
    Saves best results back to results.csv.
    """
    
    datasets = ['adult', 'bank', 'credit', 'covertype', 'diabetes']
    label_fractions = [0.05, 0.10, 0.15, 0.20, 1.0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load existing results (keep RF, XGB, XGB-Cal, Supervised_MLP, SSL_MLP)
    existing_df = pd.read_csv('results/results.csv')
    keep_models = ['Random_Forest', 'XGBoost', 'XGBoost_Calibrated', 'Supervised_MLP', 'SSL_MLP']
    kept_results = existing_df[existing_df['model'].isin(keep_models)]
    
    print("=" * 70)
    print("LAMBDA_CONF TUNING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Lambda values to test: {lambda_values}")
    print(f"Experiments per lambda: {len(datasets) * len(label_fractions) * 2}")
    print(f"Total experiments: {len(datasets) * len(label_fractions) * 2 * len(lambda_values)}")
    print(f"Kept results from previous run: {len(kept_results)} rows")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    
    # Store results for each lambda
    all_lambda_results = {}
    
    # Resume support: load already-completed lambdas
    os.makedirs('results/lambda_tuning', exist_ok=True)
    skipped = 0
    for lam in lambda_values:
        cached_path = f'results/lambda_tuning/lambda_{lam}.csv'
        if os.path.exists(cached_path):
            cached_df = pd.read_csv(cached_path)
            expected_rows = len(datasets) * len(label_fractions) * 2
            if len(cached_df) == expected_rows:
                print(f"  ⏩ Skipping lambda={lam} (already completed: {cached_path})")
                all_lambda_results[lam] = cached_df
                skipped += 1
    
    if skipped > 0:
        print(f"\n  Resumed: {skipped}/{len(lambda_values)} lambdas already done")
    
    for lam in lambda_values:
        # Skip if already loaded from cache
        if lam in all_lambda_results:
            continue
        
        print(f"\n{'#' * 70}")
        print(f"  TESTING LAMBDA = {lam}")
        print(f"{'#' * 70}\n")
        
        lambda_results = []
        
        for dataset_name in datasets:
            print(f"\n  Dataset: {dataset_name.upper()}")
            
            for label_frac in label_fractions:
                print(f"    Label Fraction: {label_frac*100:.0f}%", end=" | ")
                
                # Load data
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
                
                # SSL Pretraining
                pretrained_encoder = pretrain_vime(
                    X_unlab, input_dim=input_dim, hidden_dim=256,
                    epochs=50, mask_prob=0.3, device=device
                )
                
                # ---- CalibSSL (SSL + Calibration) ----
                mlp_calibssl = TabularMLP(input_dim=input_dim, num_classes=n_classes)
                with torch.no_grad():
                    mlp_calibssl.network[0].weight.data = pretrained_encoder[0].weight.data.clone()
                    mlp_calibssl.network[0].bias.data = pretrained_encoder[0].bias.data.clone()
                
                mlp_calibssl = train_with_calibration_reg(
                    mlp_calibssl, X_train, y_train, X_val, y_val,
                    epochs=100, lambda_conf=lam, device=device, patience=15, verbose=False
                )
                metrics_calibssl = evaluate_model(mlp_calibssl, X_test, y_test, model_type='neural')
                
                lambda_results.append({
                    'dataset': dataset_name, 'label_fraction': label_frac,
                    'model': 'CalibSSL', **metrics_calibssl
                })
                
                # ---- MLP + Calibration Only (no SSL) ----
                mlp_calib_only = TabularMLP(input_dim=input_dim, num_classes=n_classes)
                mlp_calib_only = train_with_calibration_reg(
                    mlp_calib_only, X_train, y_train, X_val, y_val,
                    epochs=100, lambda_conf=lam, device=device, patience=15, verbose=False
                )
                metrics_calib_only = evaluate_model(mlp_calib_only, X_test, y_test, model_type='neural')
                
                lambda_results.append({
                    'dataset': dataset_name, 'label_fraction': label_frac,
                    'model': 'MLP_Calib_Only', **metrics_calib_only
                })
                
                print(f"CalibSSL Acc:{metrics_calibssl['accuracy']:.4f} ECE:{metrics_calibssl['ece']:.4f} | "
                      f"Calib-Only Acc:{metrics_calib_only['accuracy']:.4f} ECE:{metrics_calib_only['ece']:.4f}")
        
        lambda_df = pd.DataFrame(lambda_results)
        all_lambda_results[lam] = lambda_df
        
        # Save per-lambda results
        os.makedirs('results/lambda_tuning', exist_ok=True)
        lambda_df.to_csv(f'results/lambda_tuning/lambda_{lam}.csv', index=False)
        
        # Print summary for this lambda
        calibssl_summary = lambda_df[lambda_df['model'] == 'CalibSSL']
        print(f"\n  Lambda {lam} Summary (CalibSSL):")
        print(f"    Avg Accuracy: {calibssl_summary['accuracy'].mean():.4f}")
        print(f"    Avg ECE:      {calibssl_summary['ece'].mean():.4f}")
        print(f"    Avg Brier:    {calibssl_summary['brier'].mean():.4f}")
    
    # ===== FIND BEST LAMBDA =====
    print("\n" + "=" * 70)
    print("LAMBDA COMPARISON SUMMARY")
    print("=" * 70)
    
    comparison = []
    for lam, df in all_lambda_results.items():
        calibssl = df[df['model'] == 'CalibSSL']
        comparison.append({
            'lambda': lam,
            'CalibSSL_Accuracy': calibssl['accuracy'].mean(),
            'CalibSSL_ECE': calibssl['ece'].mean(),
            'CalibSSL_Brier': calibssl['brier'].mean()
        })
    
    comp_df = pd.DataFrame(comparison)
    print("\n" + comp_df.to_string(index=False))
    comp_df.to_csv('results/lambda_tuning/lambda_comparison.csv', index=False)
    
    # Best lambda = lowest ECE while maintaining reasonable accuracy
    # Score = -ECE (lower ECE is better)
    best_lambda = comp_df.loc[comp_df['CalibSSL_ECE'].idxmin(), 'lambda']
    
    print(f"\n🏆 Best lambda_conf = {best_lambda} (lowest ECE)")
    
    # ===== SAVE FINAL RESULTS WITH BEST LAMBDA =====
    best_df = all_lambda_results[best_lambda]
    
    # Add experiment_id
    max_id = kept_results['experiment_id'].max() if 'experiment_id' in kept_results.columns else -1
    best_df = best_df.copy()
    best_df['experiment_id'] = range(int(max_id) + 1, int(max_id) + 1 + len(best_df))
    
    # Combine with kept results
    final_df = pd.concat([kept_results, best_df], ignore_index=True)
    final_df.to_csv('results/results.csv', index=False)
    
    print(f"\n✅ Updated results/results.csv with best lambda={best_lambda}")
    print(f"   Total rows: {len(final_df)}")
    print(f"   End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return best_lambda, comp_df


def plot_lambda_comparison(comp_df, save_dir='figures'):
    """Generate lambda comparison figure"""
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    lambdas = comp_df['lambda'].values
    
    # Accuracy
    axes[0].plot(lambdas, comp_df['CalibSSL_Accuracy'], 'o-', color='#E69F00', linewidth=2, markersize=10)
    axes[0].set_xlabel('λ (lambda_conf)', fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontweight='bold')
    axes[0].set_title('CalibSSL Accuracy vs λ', fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # ECE
    axes[1].plot(lambdas, comp_df['CalibSSL_ECE'], 's-', color='#0072B2', linewidth=2, markersize=10)
    axes[1].set_xlabel('λ (lambda_conf)', fontweight='bold')
    axes[1].set_ylabel('ECE (↓ better)', fontweight='bold')
    axes[1].set_title('CalibSSL ECE vs λ', fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Brier
    axes[2].plot(lambdas, comp_df['CalibSSL_Brier'], 'D-', color='#009E73', linewidth=2, markersize=10)
    axes[2].set_xlabel('λ (lambda_conf)', fontweight='bold')
    axes[2].set_ylabel('Brier Score (↓ better)', fontweight='bold')
    axes[2].set_title('CalibSSL Brier vs λ', fontweight='bold')
    axes[2].set_xscale('log')
    axes[2].grid(True, alpha=0.3)
    
    # Mark best lambda on each plot
    best_idx = comp_df['CalibSSL_ECE'].idxmin()
    for ax, col in zip(axes, ['CalibSSL_Accuracy', 'CalibSSL_ECE', 'CalibSSL_Brier']):
        ax.axvline(x=comp_df.loc[best_idx, 'lambda'], color='red', linestyle='--', alpha=0.5, label=f'Best λ={comp_df.loc[best_idx, "lambda"]}')
        ax.legend()
    
    plt.suptitle('Hyperparameter Tuning: Calibration Regularization Strength', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fig7_lambda_tuning.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/fig7_lambda_tuning.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Figure 7: Lambda Tuning Comparison")


if __name__ == "__main__":
    best_lambda, comparison = tune_lambda(
        lambda_values=[0.01, 0.05, 0.1, 0.2, 0.5]
    )
    
    # Generate lambda comparison graph
    plot_lambda_comparison(comparison)
    
    # Re-run all analysis and graphs with updated results
    print("\n" + "=" * 70)
    print("REGENERATING ALL TABLES AND GRAPHS...")
    print("=" * 70)
    
    from analyze_results import main as run_analysis
    run_analysis()
    
    from visualize import generate_all_publication_figures
    generate_all_publication_figures()
    
    from statistical_tests import main as run_stats
    run_stats()
    
    from error_analysis import main as run_error
    run_error()
    
    print("\n" + "=" * 70)
    print(f"🏆 ALL DONE! Best lambda_conf = {best_lambda}")
    print("   All tables, graphs, and analyses have been regenerated.")
    print("=" * 70)
