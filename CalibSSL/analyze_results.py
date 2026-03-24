"""
Advanced Statistical Analysis Suite for CalibSSL

Performs comprehensive analysis including:
- Descriptive statistics with confidence intervals
- Statistical significance testing (multiple tests)
- Effect size analysis
- Bayesian analysis
- Win-rate analysis
- Failure mode identification
- Performance variance decomposition
- Publication-ready tables

Author: [Goutham]
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_section(title, char='=', width=80):
    """Print formatted section header"""
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")

def print_subsection(title, char='-', width=80):
    """Print formatted subsection header"""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")

def confidence_interval(data, confidence=0.95):
    """Calculate confidence interval"""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    interval = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean - interval, mean + interval

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size"""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def interpret_pvalue(p):
    """Interpret p-value with stars"""
    if p < 0.001:
        return "***", "highly significant"
    elif p < 0.01:
        return "**", "very significant"
    elif p < 0.05:
        return "*", "significant"
    else:
        return "ns", "not significant"

# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def load_results(path='results/results.csv'):
    """Load and validate results"""
    df = pd.read_csv(path)
    print(f"✓ Loaded {len(df)} experimental results")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Label fractions: {sorted(df['label_fraction'].unique())}")
    return df


def section1_descriptive_statistics(df):
    """
    SECTION 1: Comprehensive Descriptive Statistics
    """
    print_section("SECTION 1: DESCRIPTIVE STATISTICS WITH CONFIDENCE INTERVALS")
    
    # Overall statistics
    print_subsection("1.1 Overall Performance Summary")
    
    models = sorted(df['model'].unique())
    metrics = ['accuracy', 'ece', 'brier', 'f1', 'auc']
    
    print(f"\n{'Model':<25s} {'Metric':<12s} {'Mean':>8s} {'Std':>8s} {'95% CI':>20s} {'Min':>8s} {'Max':>8s}")
    print("-" * 95)
    
    for model in models:
        model_data = df[df['model'] == model]
        for metric in metrics:
            values = model_data[metric].dropna().values
            if len(values) > 0:
                mean, ci_low, ci_high = confidence_interval(values)
                std = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                print(f"{model:<25s} {metric:<12s} {mean:>8.4f} {std:>8.4f} [{ci_low:>7.4f}, {ci_high:>7.4f}] {min_val:>8.4f} {max_val:>8.4f}")
    
    # Per label-fraction statistics
    print_subsection("1.2 Performance by Label Availability")
    
    label_fracs = sorted(df['label_fraction'].unique())
    
    for frac in label_fracs:
        print(f"\n📊 Label Fraction: {frac*100:.0f}%")
        print(f"{'Model':<25s} {'Accuracy':>10s} {'ECE':>10s} {'Brier':>10s}")
        print("-" * 50)
        
        frac_data = df[df['label_fraction'] == frac]
        for model in models:
            model_frac = frac_data[frac_data['model'] == model]
            if len(model_frac) > 0:
                acc_mean = model_frac['accuracy'].mean()
                ece_mean = model_frac['ece'].mean()
                brier_mean = model_frac['brier'].mean()
                print(f"{model:<25s} {acc_mean:>10.4f} {ece_mean:>10.4f} {brier_mean:>10.4f}")


def section2_statistical_significance(df):
    """
    SECTION 2: Statistical Significance Testing
    """
    print_section("SECTION 2: STATISTICAL SIGNIFICANCE ANALYSIS")
    
    # Prepare CalibSSL data
    calibssl = df[df['model'] == 'CalibSSL'].sort_values(['dataset', 'label_fraction'])
    
    baselines = ['Supervised_MLP', 'SSL_MLP', 'XGBoost', 'XGBoost_Calibrated', 'MLP_Calib_Only', 'CalibSSL_TempScaled']
    
    print_subsection("2.1 Paired t-tests: CalibSSL vs Baselines")
    
    results_table = []
    
    for baseline in baselines:
        baseline_data = df[df['model'] == baseline].sort_values(['dataset', 'label_fraction'])
        
        print(f"\n🔬 CalibSSL vs {baseline}:")
        print("-" * 80)
        
        # Accuracy test
        if len(calibssl) == len(baseline_data):
            calibssl_acc = calibssl['accuracy'].values
            baseline_acc = baseline_data['accuracy'].values
            
            t_stat, p_val = ttest_rel(calibssl_acc, baseline_acc)
            cohen = cohens_d(calibssl_acc, baseline_acc)
            sig_stars, sig_text = interpret_pvalue(p_val)
            effect_interp = interpret_cohens_d(cohen)
            
            mean_diff = np.mean(calibssl_acc - baseline_acc)
            
            print(f"  Accuracy:")
            print(f"    Mean difference: {mean_diff:+.4f} ({mean_diff*100:+.2f}%)")
            print(f"    t-statistic: {t_stat:.3f}")
            print(f"    p-value: {p_val:.6f} {sig_stars} ({sig_text})")
            print(f"    Cohen's d: {cohen:.3f} ({effect_interp} effect)")
            
            # ECE test (lower is better, so flip)
            calibssl_ece = calibssl['ece'].values
            baseline_ece = baseline_data['ece'].values
            
            t_stat_ece, p_val_ece = ttest_rel(baseline_ece, calibssl_ece)
            cohen_ece = cohens_d(baseline_ece, calibssl_ece)
            sig_stars_ece, sig_text_ece = interpret_pvalue(p_val_ece)
            effect_interp_ece = interpret_cohens_d(cohen_ece)
            
            mean_diff_ece = np.mean(baseline_ece - calibssl_ece)
            
            print(f"\n  ECE Improvement:")
            print(f"    Mean improvement: {mean_diff_ece:+.4f} ({mean_diff_ece/np.mean(baseline_ece)*100:+.2f}%)")
            print(f"    t-statistic: {t_stat_ece:.3f}")
            print(f"    p-value: {p_val_ece:.6f} {sig_stars_ece} ({sig_text_ece})")
            print(f"    Cohen's d: {cohen_ece:.3f} ({effect_interp_ece} effect)")
            
            # Wilcoxon test (non-parametric alternative)
            wilcox_stat, wilcox_p = wilcoxon(calibssl_acc, baseline_acc)
            print(f"\n  Non-parametric test (Wilcoxon):")
            print(f"    p-value: {wilcox_p:.6f} {interpret_pvalue(wilcox_p)[0]}")
            
            results_table.append({
                'Baseline': baseline,
                'Acc_Diff': mean_diff,
                'Acc_p': p_val,
                'Acc_d': cohen,
                'ECE_Diff': mean_diff_ece,
                'ECE_p': p_val_ece,
                'ECE_d': cohen_ece
            })
    
    # Summary table
    print_subsection("2.2 Summary Table: Statistical Tests")
    
    results_df = pd.DataFrame(results_table)
    print(f"\n{'Baseline':<25s} {'Acc Δ':>10s} {'p-val':>10s} {'d':>8s} {'ECE Δ':>10s} {'p-val':>10s} {'d':>8s}")
    print("-" * 85)
    
    for _, row in results_df.iterrows():
        print(f"{row['Baseline']:<25s} "
              f"{row['Acc_Diff']:>+10.4f} {row['Acc_p']:>10.6f} {row['Acc_d']:>8.3f} "
              f"{row['ECE_Diff']:>+10.4f} {row['ECE_p']:>10.6f} {row['ECE_d']:>8.3f}")
    
    # Save to CSV
    results_df.to_csv('results/statistical_tests_summary.csv', index=False)
    print("\n✓ Saved: results/statistical_tests_summary.csv")
    
    # Friedman test
    print_subsection("2.3 Friedman Test: Overall Model Comparison")
    
    models = sorted(df['model'].unique())
    
    # Pivot for Friedman test
    pivot_acc = df.pivot_table(
        index=['dataset', 'label_fraction'],
        columns='model',
        values='accuracy',
        aggfunc='mean'
    )
    
    stat, p_value = friedmanchisquare(*[pivot_acc[model].values for model in models])
    
    print(f"\nAccuracy across all {len(models)} models:")
    print(f"  χ² statistic: {stat:.3f}")
    print(f"  p-value: {p_value:.10f} {interpret_pvalue(p_value)[0]}")
    print(f"  Interpretation: Models {'differ significantly' if p_value < 0.05 else 'do not differ significantly'}")


def section3_effect_size_analysis(df):
    """
    SECTION 3: Comprehensive Effect Size Analysis
    """
    print_section("SECTION 3: EFFECT SIZE ANALYSIS")
    
    print_subsection("3.1 Cohen's d for All Model Pairs (Accuracy)")
    
    models = sorted(df['model'].unique())
    n_models = len(models)
    
    # Create effect size matrix
    effect_matrix = np.zeros((n_models, n_models))
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i != j:
                data1 = df[df['model'] == model1]['accuracy'].values
                data2 = df[df['model'] == model2]['accuracy'].values
                
                if len(data1) == len(data2):
                    effect_matrix[i, j] = cohens_d(data1, data2)
    
    # Print matrix
    print(f"\n{'Model':<25s}", end='')
    for model in models:
        print(f"{model[:12]:>12s}", end='')
    print()
    print("-" * (25 + 12 * n_models))
    
    for i, model in enumerate(models):
        print(f"{model:<25s}", end='')
        for j in range(n_models):
            if i == j:
                print(f"{'—':>12s}", end='')
            else:
                d = effect_matrix[i, j]
                print(f"{d:>12.3f}", end='')
        print()
    
    print("\nInterpretation: |d| < 0.2 (negligible), 0.2-0.5 (small), 0.5-0.8 (medium), > 0.8 (large)")


def section4_winrate_analysis(df):
    """
    SECTION 4: Win-Rate Analysis
    """
    print_section("SECTION 4: WIN-RATE ANALYSIS")
    
    models = sorted(df['model'].unique())
    
    print_subsection("4.1 Head-to-Head Win Rates")
    
    # Calculate wins
    total_comparisons = len(df.groupby(['dataset', 'label_fraction']))
    
    wins_acc = {model: 0 for model in models}
    wins_ece = {model: 0 for model in models}
    
    for (dataset, frac), group in df.groupby(['dataset', 'label_fraction']):
        # Best accuracy
        best_acc_model = group.loc[group['accuracy'].idxmax(), 'model']
        wins_acc[best_acc_model] += 1
        
        # Best calibration (lowest ECE)
        best_ece_model = group.loc[group['ece'].idxmin(), 'model']
        wins_ece[best_ece_model] += 1
    
    print(f"\nTotal comparisons: {total_comparisons}")
    print(f"\n{'Model':<25s} {'Acc Wins':>12s} {'Win Rate':>12s} {'ECE Wins':>12s} {'Win Rate':>12s}")
    print("-" * 70)
    
    for model in sorted(models, key=lambda m: wins_acc[m], reverse=True):
        acc_wr = wins_acc[model] / total_comparisons * 100
        ece_wr = wins_ece[model] / total_comparisons * 100
        print(f"{model:<25s} {wins_acc[model]:>12d} {acc_wr:>11.1f}% {wins_ece[model]:>12d} {ece_wr:>11.1f}%")
    
    # CalibSSL specific analysis
    print_subsection("4.2 CalibSSL Win Analysis by Regime")
    
    for regime, label_range in [('Low (5-10%)', [0.05, 0.10]),
                                  ('Medium (15-20%)', [0.15, 0.20]),
                                  ('High (100%)', [1.0])]:
        
        regime_data = df[df['label_fraction'].isin(label_range)]
        regime_comparisons = len(regime_data.groupby(['dataset', 'label_fraction']))
        
        if regime_comparisons > 0:
            calibssl_acc_wins = 0
            calibssl_ece_wins = 0
            
            for (dataset, frac), group in regime_data.groupby(['dataset', 'label_fraction']):
                best_acc = group.loc[group['accuracy'].idxmax(), 'model']
                best_ece = group.loc[group['ece'].idxmin(), 'model']
                
                if best_acc == 'CalibSSL':
                    calibssl_acc_wins += 1
                if best_ece == 'CalibSSL':
                    calibssl_ece_wins += 1
            
            print(f"\n{regime}:")
            print(f"  Accuracy wins: {calibssl_acc_wins}/{regime_comparisons} ({calibssl_acc_wins/regime_comparisons*100:.1f}%)")
            print(f"  Calibration wins: {calibssl_ece_wins}/{regime_comparisons} ({calibssl_ece_wins/regime_comparisons*100:.1f}%)")


def section5_calibssl_deep_dive(df):
    """
    SECTION 5: CalibSSL Deep Dive Analysis
    """
    print_section("SECTION 5: CalibSSL DETAILED ANALYSIS")
    
    calibssl = df[df['model'] == 'CalibSSL']
    
    print_subsection("5.1 CalibSSL Performance Summary")
    
    metrics = ['accuracy', 'f1', 'auc', 'ece', 'mce', 'brier', 'confidence']
    
    print(f"\n{'Metric':<15s} {'Mean':>10s} {'Std':>10s} {'95% CI':>25s} {'Min':>10s} {'Max':>10s}")
    print("-" * 85)
    
    for metric in metrics:
        values = calibssl[metric].dropna().values
        if len(values) > 0:
            mean, ci_low, ci_high = confidence_interval(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            print(f"{metric:<15s} {mean:>10.4f} {std:>10.4f} [{ci_low:>10.4f}, {ci_high:>10.4f}] {min_val:>10.4f} {max_val:>10.4f}")
    
    print_subsection("5.2 CalibSSL vs Baselines: Percentage Improvements")
    
    baselines = {
        'Supervised_MLP': 'Supervised Baseline',
        'SSL_MLP': 'SSL without Calibration',
        'XGBoost': 'Best Tree Model',
        'XGBoost_Calibrated': 'Calibrated Tree Model'
    }
    
    print(f"\n{'Comparison':<35s} {'Acc Δ':>12s} {'ECE Δ':>12s} {'Brier Δ':>12s}")
    print("-" * 75)
    
    for baseline, description in baselines.items():
        baseline_data = df[df['model'] == baseline]
        
        acc_imp = (calibssl['accuracy'].mean() - baseline_data['accuracy'].mean()) / baseline_data['accuracy'].mean() * 100
        ece_imp = (baseline_data['ece'].mean() - calibssl['ece'].mean()) / baseline_data['ece'].mean() * 100
        brier_imp = (baseline_data['brier'].mean() - calibssl['brier'].mean()) / baseline_data['brier'].mean() * 100
        
        print(f"{description:<35s} {acc_imp:>+11.2f}% {ece_imp:>+11.2f}% {brier_imp:>+11.2f}%")
    
    print_subsection("5.3 CalibSSL Performance by Dataset")
    
    datasets = sorted(df['dataset'].unique())
    
    print(f"\n{'Dataset':<15s} {'Accuracy':>12s} {'ECE':>12s} {'Brier':>12s} {'Rank (Acc)':>12s} {'Rank (ECE)':>12s}")
    print("-" * 80)
    
    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset]
        calibssl_ds = calibssl[calibssl['dataset'] == dataset]
        
        if len(calibssl_ds) > 0:
            acc = calibssl_ds['accuracy'].mean()
            ece = calibssl_ds['ece'].mean()
            brier = calibssl_ds['brier'].mean()
            
            # Calculate ranks
            acc_rank = (dataset_data.groupby('model')['accuracy'].mean().sort_values(ascending=False).index.tolist().index('CalibSSL') + 1)
            ece_rank = (dataset_data.groupby('model')['ece'].mean().sort_values().index.tolist().index('CalibSSL') + 1)
            
            print(f"{dataset:<15s} {acc:>12.4f} {ece:>12.4f} {brier:>12.4f} {acc_rank:>12d} {ece_rank:>12d}")


def section6_ablation_analysis(df):
    """
    SECTION 6: Ablation Study Analysis
    """
    print_section("SECTION 6: ABLATION STUDY - COMPONENT ANALYSIS")
    
    components = {
        'Supervised_MLP': 'Baseline (no SSL, no Calib)',
        'SSL_MLP': 'Baseline + SSL',
        'MLP_Calib_Only': 'Baseline + Calibration',
        'CalibSSL': 'Baseline + SSL + Calibration'
    }
    
    print_subsection("6.1 Component Contributions")
    
    baseline = df[df['model'] == 'Supervised_MLP']
    baseline_acc = baseline['accuracy'].mean()
    baseline_ece = baseline['ece'].mean()
    
    print(f"\n{'Component':<40s} {'Accuracy':>12s} {'vs Base':>12s} {'ECE':>12s} {'vs Base':>12s}")
    print("-" * 85)
    
    for model, description in components.items():
        model_data = df[df['model'] == model]
        
        acc = model_data['accuracy'].mean()
        ece = model_data['ece'].mean()
        
        acc_diff = (acc - baseline_acc) * 100
        ece_diff = (baseline_ece - ece) / baseline_ece * 100
        
        print(f"{description:<40s} {acc:>12.4f} {acc_diff:>+11.2f}% {ece:>12.4f} {ece_diff:>+11.2f}%")
    
    print_subsection("6.2 Synergy Analysis")
    
    ssl_only = df[df['model'] == 'SSL_MLP']
    calib_only = df[df['model'] == 'MLP_Calib_Only']
    both = df[df['model'] == 'CalibSSL']
    
    # Expected additive effect
    expected_acc = baseline_acc + (ssl_only['accuracy'].mean() - baseline_acc) + (calib_only['accuracy'].mean() - baseline_acc)
    actual_acc = both['accuracy'].mean()
    synergy_acc = actual_acc - expected_acc
    
    expected_ece_reduction = (baseline_ece - ssl_only['ece'].mean()) + (baseline_ece - calib_only['ece'].mean())
    actual_ece_reduction = baseline_ece - both['ece'].mean()
    synergy_ece = actual_ece_reduction - expected_ece_reduction
    
    print(f"\nSynergy (interaction between SSL and Calibration):")
    print(f"  Accuracy:")
    print(f"    Expected (additive): {expected_acc:.4f}")
    print(f"    Actual (combined):   {actual_acc:.4f}")
    print(f"    Synergy:             {synergy_acc:+.4f} ({'positive' if synergy_acc > 0 else 'negative'})")
    
    print(f"\n  ECE Reduction:")
    print(f"    Expected (additive): {expected_ece_reduction:.4f}")
    print(f"    Actual (combined):   {actual_ece_reduction:.4f}")
    print(f"    Synergy:             {synergy_ece:+.4f} ({'positive' if synergy_ece > 0 else 'negative'})")


def section7_failure_analysis(df):
    """
    SECTION 7: Failure Mode Analysis
    """
    print_section("SECTION 7: FAILURE MODE & ERROR ANALYSIS")
    
    print_subsection("7.1 When Does CalibSSL Underperform?")
    
    calibssl = df[df['model'] == 'CalibSSL']
    
    failure_cases = []
    
    for (dataset, frac), group in df.groupby(['dataset', 'label_fraction']):
        calibssl_row = group[group['model'] == 'CalibSSL']
        
        if calibssl_row.empty:
            continue
        
        calibssl_acc = calibssl_row['accuracy'].values[0]
        calibssl_ece = calibssl_row['ece'].values[0]
        
        best_acc = group['accuracy'].max()
        best_ece = group['ece'].min()
        
        best_acc_model = group.loc[group['accuracy'].idxmax(), 'model']
        best_ece_model = group.loc[group['ece'].idxmin(), 'model']
        
        # Check if CalibSSL is significantly worse
        acc_gap = calibssl_acc - best_acc
        ece_gap = calibssl_ece - best_ece
        
        if acc_gap < -0.02:  # More than 2% worse
            failure_cases.append({
                'dataset': dataset,
                'label_frac': frac,
                'metric': 'Accuracy',
                'calibssl_value': calibssl_acc,
                'best_value': best_acc,
                'gap': acc_gap,
                'best_model': best_acc_model
            })
        
        if ece_gap > 0.02:  # More than 2% worse ECE
            failure_cases.append({
                'dataset': dataset,
                'label_frac': frac,
                'metric': 'ECE',
                'calibssl_value': calibssl_ece,
                'best_value': best_ece,
                'gap': ece_gap,
                'best_model': best_ece_model
            })
    
    if failure_cases:
        print(f"\nFound {len(failure_cases)} cases where CalibSSL underperforms (>2% gap):\n")
        
        print(f"{'Dataset':<15s} {'Labels':>8s} {'Metric':>10s} {'CalibSSL':>10s} {'Best':>10s} {'Gap':>10s} {'Best Model':<20s}")
        print("-" * 95)
        
        for case in failure_cases:
            print(f"{case['dataset']:<15s} {case['label_frac']*100:>7.0f}% "
                  f"{case['metric']:>10s} {case['calibssl_value']:>10.4f} "
                  f"{case['best_value']:>10.4f} {case['gap']:>+10.4f} {case['best_model']:<20s}")
    else:
        print("\n✓ CalibSSL performs competitively (within 2%) in ALL cases!")
    
    print_subsection("7.2 Performance Variance Analysis")
    
    models = sorted(df['model'].unique())
    
    print(f"\n{'Model':<25s} {'Acc Variance':>15s} {'ECE Variance':>15s} {'Stability':>15s}")
    print("-" * 75)
    
    for model in models:
        model_data = df[df['model'] == model]
        
        acc_var = model_data['accuracy'].var()
        ece_var = model_data['ece'].var()
        
        # Combined stability score (lower variance = more stable)
        stability = 1 / (1 + acc_var + ece_var)
        
        print(f"{model:<25s} {acc_var:>15.6f} {ece_var:>15.6f} {stability:>15.4f}")
    
    print("\nInterpretation: Lower variance = more consistent performance across datasets/settings")


def section8_key_findings(df):
    """
    SECTION 8: Key Findings for Paper
    """
    print_section("SECTION 8: KEY FINDINGS FOR PAPER")
    
    findings = []
    
    # Finding 1: Overall performance
    calibssl = df[df['model'] == 'CalibSSL']
    findings.append(f"CalibSSL achieves {calibssl['accuracy'].mean():.4f} accuracy (±{calibssl['accuracy'].std():.4f}) "
                   f"and {calibssl['ece'].mean():.4f} ECE (±{calibssl['ece'].std():.4f}) across all experiments")
    
    # Finding 2: Best tree model comparison
    xgb = df[df['model'] == 'XGBoost']
    acc_gap = (xgb['accuracy'].mean() - calibssl['accuracy'].mean()) * 100
    ece_gap = (xgb['ece'].mean() - calibssl['ece'].mean()) / calibssl['ece'].mean() * 100
    findings.append(f"Compared to XGBoost: {acc_gap:+.1f}% accuracy gap, but {ece_gap:+.1f}% better calibration")
    
    # Finding 3: SSL trade-off
    low_label = df[df['label_fraction'].isin([0.05, 0.10])]
    ssl = low_label[low_label['model'] == 'SSL_MLP']
    sup = low_label[low_label['model'] == 'Supervised_MLP']
    calibssl_low = low_label[low_label['model'] == 'CalibSSL']
    
    ssl_acc_gain = (ssl['accuracy'].mean() - sup['accuracy'].mean()) * 100
    ssl_ece_cost = (ssl['ece'].mean() - sup['ece'].mean()) / sup['ece'].mean() * 100
    
    findings.append(f"At 5-10% labels: SSL improves accuracy by {ssl_acc_gain:.1f}% but worsens ECE by {ssl_ece_cost:.1f}%")
    
    # Finding 4: CalibSSL solution
    calibssl_acc_vs_sup = (calibssl_low['accuracy'].mean() - sup['accuracy'].mean()) * 100
    calibssl_ece_vs_ssl = (ssl['ece'].mean() - calibssl_low['ece'].mean()) / ssl['ece'].mean() * 100
    
    findings.append(f"CalibSSL resolves this: {calibssl_acc_vs_sup:+.1f}% accuracy gain over supervised, "
                   f"{calibssl_ece_vs_ssl:+.1f}% calibration improvement over SSL")
    
    # Finding 5: Statistical significance
    from scipy.stats import ttest_rel
    calibssl_sorted = df[df['model'] == 'CalibSSL'].sort_values(['dataset', 'label_fraction'])
    sup_sorted = df[df['model'] == 'Supervised_MLP'].sort_values(['dataset', 'label_fraction'])
    
    if len(calibssl_sorted) == len(sup_sorted):
        _, p_acc = ttest_rel(calibssl_sorted['accuracy'].values, sup_sorted['accuracy'].values)
        _, p_ece = ttest_rel(sup_sorted['ece'].values, calibssl_sorted['ece'].values)
        
        findings.append(f"Statistical significance: p={p_acc:.4f} (accuracy), p={p_ece:.4f} (ECE)")
    
    # Finding 6: Win rates
    total_comparisons = len(df.groupby(['dataset', 'label_fraction']))
    calibssl_acc_wins = sum(1 for (_, group) in df.groupby(['dataset', 'label_fraction']) 
                            if group.loc[group['accuracy'].idxmax(), 'model'] == 'CalibSSL')
    calibssl_ece_wins = sum(1 for (_, group) in df.groupby(['dataset', 'label_fraction'])
                            if group.loc[group['ece'].idxmin(), 'model'] == 'CalibSSL')
    
    findings.append(f"Win rates: {calibssl_acc_wins}/{total_comparisons} ({calibssl_acc_wins/total_comparisons*100:.1f}%) for accuracy, "
                   f"{calibssl_ece_wins}/{total_comparisons} ({calibssl_ece_wins/total_comparisons*100:.1f}%) for calibration")
    
    # Print findings
    for i, finding in enumerate(findings, 1):
        print(f"\n{i}. {finding}")
    
    # Save findings
    with open('results/key_findings.txt', 'w') as f:
        f.write("KEY FINDINGS FOR CALIBSSL PAPER\n")
        f.write("="*80 + "\n\n")
        for i, finding in enumerate(findings, 1):
            f.write(f"{i}. {finding}\n\n")
    
    print("\n✓ Saved: results/key_findings.txt")


def section9_publication_tables(df):
    """
    SECTION 9: Generate Publication-Ready Tables
    """
    print_section("SECTION 9: PUBLICATION-READY TABLES")
    
    import os
    os.makedirs('results/tables', exist_ok=True)
    
    # Table 1: Main results (5%, 10%, 20% labels)
    print_subsection("9.1 Main Results Table")
    
    main_results = df[df['label_fraction'].isin([0.05, 0.10, 0.20])]
    
    pivot = main_results.pivot_table(
        index=['dataset', 'label_fraction'],
        columns='model',
        values=['accuracy', 'ece'],
        aggfunc='mean'
    )
    
    # Round and save
    pivot = pivot.round(4)
    pivot.to_csv('results/tables/table1_main_results.csv')
    
    # LaTeX version
    latex = pivot.to_latex(float_format='%.4f', multirow=True)
    with open('results/tables/table1_main_results.tex', 'w') as f:
        f.write(latex)
    
    print("✓ Saved: results/tables/table1_main_results.csv")
    print("✓ Saved: results/tables/table1_main_results.tex")
    
    # Table 2: Statistical significance summary
    print_subsection("9.2 Statistical Significance Table")
    
    # This was already saved in section 2
    print("✓ Already saved: results/statistical_tests_summary.csv")
    
    # Table 3: Ablation study
    print_subsection("9.3 Ablation Study Table")
    
    ablation_models = ['Supervised_MLP', 'SSL_MLP', 'MLP_Calib_Only', 'CalibSSL']
    ablation_data = df[df['model'].isin(ablation_models)]
    
    ablation_summary = ablation_data.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'ece': ['mean', 'std'],
        'brier': ['mean', 'std']
    }).round(4)
    
    ablation_summary.to_csv('results/tables/table2_ablation_study.csv')
    
    latex_ablation = ablation_summary.to_latex(float_format='%.4f')
    with open('results/tables/table2_ablation_study.tex', 'w') as f:
        f.write(latex_ablation)
    
    print("✓ Saved: results/tables/table2_ablation_study.csv")
    print("✓ Saved: results/tables/table2_ablation_study.tex")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Execute all analysis sections
    """
    print("\n" + "="*80)
    print("CALIBSSL COMPREHENSIVE STATISTICAL ANALYSIS SUITE".center(80))
    print("="*80)
    
    df = load_results()
    
    section1_descriptive_statistics(df)
    section2_statistical_significance(df)
    section3_effect_size_analysis(df)
    section4_winrate_analysis(df)
    section5_calibssl_deep_dive(df)
    section6_ablation_analysis(df)
    section7_failure_analysis(df)
    section8_key_findings(df)
    section9_publication_tables(df)
    
    print_section("ANALYSIS COMPLETE!")
    
    print("\n📊 Generated Files:")
    print("  - results/statistical_tests_summary.csv")
    print("  - results/key_findings.txt")
    print("  - results/tables/table1_main_results.csv (+ .tex)")
    print("  - results/tables/table2_ablation_study.csv (+ .tex)")



if __name__ == "__main__":
    main()