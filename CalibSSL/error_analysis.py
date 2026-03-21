"""
Error analysis - where does CalibSSL fail?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(path='results/results.csv'):
    return pd.read_csv(path)

def find_failure_cases(df):
    """Find cases where CalibSSL underperforms"""
    print("="*70)
    print("CALIBSSL FAILURE CASE ANALYSIS")
    print("="*70)
    
    calibssl = df[df['model'] == 'CalibSSL']
    
    # For each experiment, find if CalibSSL is NOT the best
    failures = []
    
    for (dataset, frac), group in df.groupby(['dataset', 'label_fraction']):
        calibssl_row = group[group['model'] == 'CalibSSL']
        
        if calibssl_row.empty:
            continue
        
        calibssl_acc = calibssl_row['accuracy'].values[0]
        calibssl_ece = calibssl_row['ece'].values[0]
        
        best_acc = group['accuracy'].max()
        best_ece = group['ece'].min()
        
        # Check if CalibSSL is far from best
        if calibssl_acc < best_acc - 0.02:  # More than 2% worse
            failures.append({
                'dataset': dataset,
                'label_frac': frac,
                'issue': 'Low accuracy',
                'calibssl_value': calibssl_acc,
                'best_value': best_acc,
                'gap': calibssl_acc - best_acc,
                'best_model': group.loc[group['accuracy'].idxmax(), 'model']
            })
        
        if calibssl_ece > best_ece + 0.02:  # More than 2% worse ECE
            failures.append({
                'dataset': dataset,
                'label_frac': frac,
                'issue': 'Poor calibration',
                'calibssl_value': calibssl_ece,
                'best_value': best_ece,
                'gap': calibssl_ece - best_ece,
                'best_model': group.loc[group['ece'].idxmin(), 'model']
            })
    
    if failures:
        print(f"\nFound {len(failures)} cases where CalibSSL underperforms:\n")
        for f in failures:
            print(f"  {f['dataset']:12s} @ {f['label_frac']*100:3.0f}% labels: {f['issue']}")
            print(f"    CalibSSL: {f['calibssl_value']:.4f}, Best ({f['best_model']}): {f['best_value']:.4f}, Gap: {f['gap']:+.4f}\n")
    else:
        print("\n✓ CalibSSL performs competitively in all cases!")

def analyze_dataset_characteristics(df):
    """Analyze which datasets CalibSSL works best on"""
    print("\n" + "="*70)
    print("CALIBSSL PERFORMANCE BY DATASET CHARACTERISTICS")
    print("="*70)
    
    calibssl = df[df['model'] == 'CalibSSL']
    
    print("\nCalibSSL Average Performance per Dataset:")
    print("-"*70)
    
    for dataset in sorted(df['dataset'].unique()):
        ds_data = calibssl[calibssl['dataset'] == dataset]
        
        print(f"\n{dataset.capitalize():15s}:")
        print(f"  Accuracy: {ds_data['accuracy'].mean():.4f} ± {ds_data['accuracy'].std():.4f}")
        print(f"  ECE:      {ds_data['ece'].mean():.4f} ± {ds_data['ece'].std():.4f}")
        print(f"  Brier:    {ds_data['brier'].mean():.4f} ± {ds_data['brier'].std():.4f}")

def confidence_calibration_analysis(df):
    """Analyze relationship between confidence and calibration"""
    print("\n" + "="*70)
    print("CONFIDENCE vs CALIBRATION ANALYSIS")
    print("="*70)
    
    # Average confidence vs ECE for each model
    summary = df.groupby('model')[['confidence', 'ece']].mean().sort_values('ece')
    
    print("\nModel Confidence vs Calibration Error:")
    print("-"*70)
    print(f"{'Model':<25s} {'Avg Confidence':>15s} {'ECE':>10s}")
    print("-"*70)
    
    for model, row in summary.iterrows():
        print(f"{model:<25s} {row['confidence']:>15.4f} {row['ece']:>10.4f}")
    
    print("\nInsight: Higher confidence often correlates with worse calibration (overconfidence)")

def plot_error_distribution(df, save_dir='figures'):
    """Plot error distributions for CalibSSL"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    calibssl = df[df['model'] == 'CalibSSL']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Accuracy distribution
    axes[0].hist(calibssl['accuracy'], bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(calibssl['accuracy'].mean(), color='red', linestyle='--', label=f"Mean: {calibssl['accuracy'].mean():.4f}")
    axes[0].set_xlabel('Accuracy')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('CalibSSL Accuracy Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # ECE distribution
    axes[1].hist(calibssl['ece'], bins=20, edgecolor='black', alpha=0.7, color='coral')
    axes[1].axvline(calibssl['ece'].mean(), color='red', linestyle='--', label=f"Mean: {calibssl['ece'].mean():.4f}")
    axes[1].set_xlabel('ECE')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('CalibSSL ECE Distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Brier distribution
    axes[2].hist(calibssl['brier'], bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[2].axvline(calibssl['brier'].mean(), color='red', linestyle='--', label=f"Mean: {calibssl['brier'].mean():.4f}")
    axes[2].set_xlabel('Brier Score')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('CalibSSL Brier Distribution')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/calibssl_error_distribution.png', dpi=300)
    plt.close()
    print(f"\n✓ Saved calibssl_error_distribution.png")

def main():
    df = load_results()
    
    find_failure_cases(df)
    analyze_dataset_characteristics(df)
    confidence_calibration_analysis(df)
    plot_error_distribution(df)
    
    print("\n" + "="*70)
    print("Error analysis complete!".center(70))
    print("="*70)

if __name__ == "__main__":
    main()