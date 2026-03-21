"""
Statistical significance tests for CalibSSL vs baselines
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel, friedmanchisquare

def load_results(path='results/results.csv'):
    return pd.read_csv(path)

def paired_t_test(df, model1, model2, metric='accuracy'):
    """
    Paired t-test comparing two models
    Null hypothesis: No difference between models
    """
    # Get paired samples (same dataset, same label fraction)
    m1_data = df[df['model'] == model1].sort_values(['dataset', 'label_fraction'])
    m2_data = df[df['model'] == model2].sort_values(['dataset', 'label_fraction'])
    
    # Ensure same experiments
    m1_values = m1_data[metric].values
    m2_values = m2_data[metric].values
    
    if len(m1_values) != len(m2_values):
        print(f"Warning: Different number of samples for {model1} and {model2}")
        return None
    
    # Paired t-test
    t_stat, p_value = ttest_rel(m1_values, m2_values)
    
    # Effect size (Cohen's d)
    diff = m1_values - m2_values
    cohen_d = diff.mean() / diff.std()
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_diff': diff.mean(),
        'cohen_d': cohen_d,
        'significant': p_value < 0.05
    }

def wilcoxon_test(df, model1, model2, metric='accuracy'):
    """
    Wilcoxon signed-rank test (non-parametric alternative to t-test)
    """
    m1_data = df[df['model'] == model1].sort_values(['dataset', 'label_fraction'])
    m2_data = df[df['model'] == model2].sort_values(['dataset', 'label_fraction'])
    
    m1_values = m1_data[metric].values
    m2_values = m2_data[metric].values
    
    if len(m1_values) != len(m2_values):
        return None
    
    stat, p_value = wilcoxon(m1_values, m2_values)
    
    return {
        'statistic': stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def print_header(title):
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70)

def test_calibssl_vs_baselines(df):
    """Test CalibSSL against all baselines"""
    print_header("STATISTICAL SIGNIFICANCE TESTS")
    
    baselines = ['Supervised_MLP', 'SSL_MLP', 'XGBoost', 'XGBoost_Calibrated', 'MLP_Calib_Only']
    
    print("\nCalibSSL vs Baselines - Paired t-tests:")
    print("-"*70)
    
    for baseline in baselines:
        print(f"\nCalibSSL vs {baseline}:")
        
        # Accuracy test
        acc_test = paired_t_test(df, 'CalibSSL', baseline, metric='accuracy')
        if acc_test:
            print(f"  Accuracy:")
            print(f"    Mean diff: {acc_test['mean_diff']:+.4f}")
            print(f"    t-stat: {acc_test['t_statistic']:.3f}")
            print(f"    p-value: {acc_test['p_value']:.4f} {'***' if acc_test['p_value'] < 0.001 else '**' if acc_test['p_value'] < 0.01 else '*' if acc_test['p_value'] < 0.05 else 'ns'}")
            print(f"    Cohen's d: {acc_test['cohen_d']:.3f}")
        
        # ECE test (lower is better, so flip comparison)
        ece_test = paired_t_test(df, baseline, 'CalibSSL', metric='ece')
        if ece_test:
            print(f"  ECE improvement:")
            print(f"    Mean diff: {ece_test['mean_diff']:+.4f}")
            print(f"    t-stat: {ece_test['t_statistic']:.3f}")
            print(f"    p-value: {ece_test['p_value']:.4f} {'***' if ece_test['p_value'] < 0.001 else '**' if ece_test['p_value'] < 0.01 else '*' if ece_test['p_value'] < 0.05 else 'ns'}")

def test_low_label_regime(df):
    """Test specifically in low-label regime (5%, 10%)"""
    print_header("LOW-LABEL REGIME SIGNIFICANCE (5% and 10%)")
    
    low_label = df[df['label_fraction'].isin([0.05, 0.10])]
    
    baselines = ['Supervised_MLP', 'SSL_MLP', 'XGBoost_Calibrated']
    
    for baseline in baselines:
        print(f"\nCalibSSL vs {baseline} (Low-Label Only):")
        
        acc_test = paired_t_test(low_label, 'CalibSSL', baseline, metric='accuracy')
        ece_test = paired_t_test(low_label, baseline, 'CalibSSL', metric='ece')
        
        if acc_test:
            sig = '✓ SIGNIFICANT' if acc_test['significant'] else '✗ Not significant'
            print(f"  Accuracy: {acc_test['mean_diff']:+.4f} (p={acc_test['p_value']:.4f}) {sig}")
        
        if ece_test:
            sig = '✓ SIGNIFICANT' if ece_test['significant'] else '✗ Not significant'
            print(f"  ECE: {ece_test['mean_diff']:+.4f} (p={ece_test['p_value']:.4f}) {sig}")

def friedman_test_all_models(df):
    """
    Friedman test - non-parametric test for multiple related samples
    Tests if there are differences among all models
    """
    print_header("FRIEDMAN TEST - OVERALL MODEL COMPARISON")
    
    # Get data for each model (same experimental conditions)
    models = sorted(df['model'].unique())
    
    # Create matrix: rows = experiments, columns = models
    pivot = df.pivot_table(
        index=['dataset', 'label_fraction'],
        columns='model',
        values='accuracy',
        aggfunc='mean'
    )
    
    # Friedman test
    stat, p_value = friedmanchisquare(*[pivot[model].values for model in models])
    
    print(f"\nAccuracy across all models:")
    print(f"  χ² statistic: {stat:.3f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Interpretation: {'Models are significantly different' if p_value < 0.05 else 'No significant difference'}")
    
    # Same for ECE
    pivot_ece = df.pivot_table(
        index=['dataset', 'label_fraction'],
        columns='model',
        values='ece',
        aggfunc='mean'
    )
    
    stat_ece, p_value_ece = friedmanchisquare(*[pivot_ece[model].values for model in models])
    
    print(f"\nECE across all models:")
    print(f"  χ² statistic: {stat_ece:.3f}")
    print(f"  p-value: {p_value_ece:.6f}")
    print(f"  Interpretation: {'Models are significantly different' if p_value_ece < 0.05 else 'No significant difference'}")

def save_significance_table(df, save_dir='results'):
    """Save significance test results as table"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    baselines = ['Supervised_MLP', 'SSL_MLP', 'XGBoost', 'XGBoost_Calibrated']
    
    results = []
    
    for baseline in baselines:
        acc_test = paired_t_test(df, 'CalibSSL', baseline, metric='accuracy')
        ece_test = paired_t_test(df, baseline, 'CalibSSL', metric='ece')
        
        if acc_test and ece_test:
            results.append({
                'Baseline': baseline,
                'Acc_MeanDiff': acc_test['mean_diff'],
                'Acc_pValue': acc_test['p_value'],
                'Acc_Significant': acc_test['significant'],
                'ECE_MeanDiff': ece_test['mean_diff'],
                'ECE_pValue': ece_test['p_value'],
                'ECE_Significant': ece_test['significant']
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{save_dir}/significance_tests.csv', index=False)
    print(f"\n✓ Saved significance_tests.csv")

def main():
    df = load_results()
    
    test_calibssl_vs_baselines(df)
    test_low_label_regime(df)
    friedman_test_all_models(df)
    save_significance_table(df)
    
    print("\n" + "="*70)
    print("Statistical testing complete!".center(70))
    print("="*70)

if __name__ == "__main__":
    main()