"""
Publication-Grade Visualization Suite for CalibSSL

Creates 15+ publication-ready figures:
- Multi-panel comparisons
- Heatmaps with statistical annotations
- Reliability diagrams
- Pareto frontiers
- Win-rate matrices
- And more...

Author: [Your Name]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 8
})

# Color palette (colorblind-friendly)
COLORS = {
    'CalibSSL': '#E69F00',           # Orange (our method)
    'SSL_MLP': '#56B4E9',            # Sky blue
    'Supervised_MLP': '#999999',     # Gray
    'MLP_Calib_Only': '#CC79A7',     # Purple
    'XGBoost': '#009E73',            # Green
    'XGBoost_Calibrated': '#0072B2', # Dark blue
    'Random_Forest': '#D55E00',      # Red-orange
    'CalibSSL_TempScaled': '#F0E442' # Yellow
}

# Model display names
MODEL_NAMES = {
    'CalibSSL': 'CalibSSL (Ours)',
    'SSL_MLP': 'SSL-MLP',
    'Supervised_MLP': 'Supervised',
    'MLP_Calib_Only': 'Calib-Only',
    'XGBoost': 'XGBoost',
    'XGBoost_Calibrated': 'XGBoost-Cal',
    'Random_Forest': 'Random Forest',
    'CalibSSL_TempScaled': 'CalibSSL (TS)'
}

def load_results(path='results/results.csv'):
    """Load and validate experimental results"""
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} experiments")
    print(f"  Datasets: {df['dataset'].nunique()}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Label fractions: {sorted(df['label_fraction'].unique())}")
    return df


# ============================================================================
# FIGURE 1: COMPREHENSIVE OVERVIEW (Main Paper Figure)
# ============================================================================

def create_figure1_comprehensive_overview(df, save_dir='figures'):
    """
    Main paper figure: 4-panel overview showing:
    - Accuracy vs label fraction (all datasets)
    - ECE vs label fraction (all datasets)
    - Accuracy-Calibration scatter (low-label regime)
    - Performance improvement heatmap
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Accuracy vs Label Fraction (Average across datasets)
    ax1 = fig.add_subplot(gs[0, 0])
    avg_acc = df.groupby(['model', 'label_fraction'])['accuracy'].agg(['mean', 'std']).reset_index()
    
    for model in sorted(df['model'].unique()):
        data = avg_acc[avg_acc['model'] == model].sort_values('label_fraction')
        ax1.plot(data['label_fraction'] * 100, data['mean'], 
                marker='o', label=MODEL_NAMES.get(model, model),
                color=COLORS.get(model, 'gray'), linewidth=2.5, markersize=8)
        ax1.fill_between(data['label_fraction'] * 100,
                         data['mean'] - data['std'],
                         data['mean'] + data['std'],
                         alpha=0.15, color=COLORS.get(model, 'gray'))
    
    ax1.set_xlabel('Label Fraction (%)', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('(A) Accuracy vs Label Availability', fontweight='bold', pad=10)
    ax1.legend(loc='lower right', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.65, 0.88])
    
    # Panel B: ECE vs Label Fraction
    ax2 = fig.add_subplot(gs[0, 1])
    avg_ece = df.groupby(['model', 'label_fraction'])['ece'].agg(['mean', 'std']).reset_index()
    
    for model in sorted(df['model'].unique()):
        data = avg_ece[avg_ece['model'] == model].sort_values('label_fraction')
        ax2.plot(data['label_fraction'] * 100, data['mean'],
                marker='o', label=MODEL_NAMES.get(model, model),
                color=COLORS.get(model, 'gray'), linewidth=2.5, markersize=8)
        ax2.fill_between(data['label_fraction'] * 100,
                         data['mean'] - data['std'],
                         data['mean'] + data['std'],
                         alpha=0.15, color=COLORS.get(model, 'gray'))
    
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='ECE=0.05 threshold')
    ax2.set_xlabel('Label Fraction (%)', fontweight='bold')
    ax2.set_ylabel('Expected Calibration Error (ECE)', fontweight='bold')
    ax2.set_title('(B) Calibration Error vs Label Availability', fontweight='bold', pad=10)
    ax2.legend(loc='upper right', framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Accuracy-Calibration Trade-off (Scatter, 5-10% labels)
    ax3 = fig.add_subplot(gs[1, 0])
    low_label = df[df['label_fraction'].isin([0.05, 0.10])]
    
    for model in sorted(df['model'].unique()):
        data = low_label[low_label['model'] == model]
        ax3.scatter(data['ece'], data['accuracy'],
                   label=MODEL_NAMES.get(model, model),
                   color=COLORS.get(model, 'gray'),
                   s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add Pareto frontier
    pareto_points = []
    for _, row in low_label.iterrows():
        is_pareto = True
        for _, other_row in low_label.iterrows():
            if (other_row['accuracy'] > row['accuracy'] and other_row['ece'] <= row['ece']) or \
               (other_row['accuracy'] >= row['accuracy'] and other_row['ece'] < row['ece']):
                is_pareto = False
                break
        if is_pareto:
            pareto_points.append(row)
    
    if pareto_points:
        pareto_df = pd.DataFrame(pareto_points).sort_values('ece')
        ax3.plot(pareto_df['ece'], pareto_df['accuracy'],
                'k--', alpha=0.4, linewidth=2, label='Pareto Frontier')
    
    ax3.axvline(x=0.05, color='red', linestyle='--', alpha=0.3)
    ax3.set_xlabel('ECE (Calibration Error) ← Better', fontweight='bold')
    ax3.set_ylabel('Accuracy (Higher is Better) ↑', fontweight='bold')
    ax3.set_title('(C) Accuracy-Calibration Trade-off (5-10% Labels)', fontweight='bold', pad=10)
    ax3.legend(loc='lower left', framealpha=0.95, fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Relative Performance Heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate percentage improvement over supervised baseline
    supervised_acc = df[df['model'] == 'Supervised_MLP'].groupby(['dataset', 'label_fraction'])['accuracy'].mean()
    
    improvement_data = []
    for model in ['CalibSSL', 'SSL_MLP', 'XGBoost', 'XGBoost_Calibrated']:
        model_data = df[df['model'] == model].groupby(['dataset', 'label_fraction'])['accuracy'].mean()
        improvement = ((model_data - supervised_acc) / supervised_acc * 100).reset_index()
        improvement['model'] = MODEL_NAMES.get(model, model)
        improvement_data.append(improvement)
    
    improvement_df = pd.concat(improvement_data)
    pivot = improvement_df.pivot_table(
        index='model',
        columns=['dataset', 'label_fraction'],
        values='accuracy'
    )
    
    # Create column labels
    col_labels = [f"{ds[:3]}\n{int(frac*100)}%" for ds, frac in pivot.columns]
    
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Accuracy Improvement (%)\nover Supervised'},
                ax=ax4, xticklabels=col_labels, linewidths=0.5)
    ax4.set_title('(D) Relative Performance vs Supervised Baseline', fontweight='bold', pad=10)
    ax4.set_xlabel('Dataset & Label Fraction', fontweight='bold')
    ax4.set_ylabel('Model', fontweight='bold')
    
    plt.suptitle('CalibSSL: Comprehensive Performance Overview',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(f'{save_dir}/fig1_comprehensive_overview.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/fig1_comprehensive_overview.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Figure 1: Comprehensive Overview")


# ============================================================================
# FIGURE 2: PER-DATASET DETAILED ANALYSIS
# ============================================================================

def create_figure2_per_dataset_analysis(df, save_dir='figures'):
    """
    5x2 grid showing accuracy and ECE for each dataset
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    datasets = sorted(df['dataset'].unique())
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(n_datasets, 2, figsize=(14, 3*n_datasets))
    
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    for idx, dataset in enumerate(datasets):
        data = df[df['dataset'] == dataset]
        
        # Left: Accuracy
        for model in sorted(data['model'].unique()):
            model_data = data[data['model'] == model].sort_values('label_fraction')
            axes[idx, 0].plot(model_data['label_fraction'] * 100,
                            model_data['accuracy'],
                            marker='o', label=MODEL_NAMES.get(model, model),
                            color=COLORS.get(model, 'gray'),
                            linewidth=2, markersize=6)
        
        axes[idx, 0].set_ylabel('Accuracy', fontweight='bold')
        axes[idx, 0].set_title(f'{dataset.capitalize()} - Accuracy', fontweight='bold')
        axes[idx, 0].legend(fontsize=7, loc='best')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Right: ECE
        for model in sorted(data['model'].unique()):
            model_data = data[data['model'] == model].sort_values('label_fraction')
            axes[idx, 1].plot(model_data['label_fraction'] * 100,
                            model_data['ece'],
                            marker='o', label=MODEL_NAMES.get(model, model),
                            color=COLORS.get(model, 'gray'),
                            linewidth=2, markersize=6)
        
        axes[idx, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.3)
        axes[idx, 1].set_ylabel('ECE', fontweight='bold')
        axes[idx, 1].set_title(f'{dataset.capitalize()} - Calibration', fontweight='bold')
        axes[idx, 1].legend(fontsize=7, loc='best')
        axes[idx, 1].grid(True, alpha=0.3)
        
        # Only add x-label to bottom row
        if idx == n_datasets - 1:
            axes[idx, 0].set_xlabel('Label Fraction (%)', fontweight='bold')
            axes[idx, 1].set_xlabel('Label Fraction (%)', fontweight='bold')
    
    plt.suptitle('Per-Dataset Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(f'{save_dir}/fig2_per_dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/fig2_per_dataset_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Figure 2: Per-Dataset Analysis")


# ============================================================================
# FIGURE 3: WIN-RATE MATRIX & STATISTICAL DOMINANCE
# ============================================================================

def create_figure3_winrate_matrix(df, save_dir='figures'):
    """
    Win-rate matrix showing how often each model beats others
    Plus statistical significance annotations
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    models = sorted(df['model'].unique())
    n_models = len(models)
    
    # Calculate win rates for accuracy
    win_matrix_acc = np.zeros((n_models, n_models))
    win_matrix_ece = np.zeros((n_models, n_models))
    
    for (dataset, frac), group in df.groupby(['dataset', 'label_fraction']):
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i != j:
                    acc1 = group[group['model'] == model1]['accuracy'].values[0]
                    acc2 = group[group['model'] == model2]['accuracy'].values[0]
                    
                    ece1 = group[group['model'] == model1]['ece'].values[0]
                    ece2 = group[group['model'] == model2]['ece'].values[0]
                    
                    if acc1 > acc2:
                        win_matrix_acc[i, j] += 1
                    
                    if ece1 < ece2:  # Lower ECE is better
                        win_matrix_ece[i, j] += 1
    
    # Convert to percentages
    total_comparisons = len(df.groupby(['dataset', 'label_fraction']))
    win_matrix_acc = (win_matrix_acc / total_comparisons) * 100
    win_matrix_ece = (win_matrix_ece / total_comparisons) * 100
    
    # Plot accuracy win rates
    model_labels = [MODEL_NAMES.get(m, m) for m in models]
    
    im1 = axes[0].imshow(win_matrix_acc, cmap='RdYlGn', vmin=0, vmax=100)
    axes[0].set_xticks(np.arange(n_models))
    axes[0].set_yticks(np.arange(n_models))
    axes[0].set_xticklabels(model_labels, rotation=45, ha='right')
    axes[0].set_yticklabels(model_labels)
    
    # Add text annotations
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                text = axes[0].text(j, i, f'{win_matrix_acc[i, j]:.0f}%',
                                   ha="center", va="center",
                                   color="white" if win_matrix_acc[i, j] > 50 else "black",
                                   fontsize=9, fontweight='bold')
    
    axes[0].set_title('(A) Accuracy Win Rate Matrix\n(Row beats Column)', fontweight='bold', pad=10)
    axes[0].set_xlabel('Model (Column)', fontweight='bold')
    axes[0].set_ylabel('Model (Row)', fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Win Rate (%)', fontweight='bold')
    
    # Plot ECE win rates
    im2 = axes[1].imshow(win_matrix_ece, cmap='RdYlGn', vmin=0, vmax=100)
    axes[1].set_xticks(np.arange(n_models))
    axes[1].set_yticks(np.arange(n_models))
    axes[1].set_xticklabels(model_labels, rotation=45, ha='right')
    axes[1].set_yticklabels(model_labels)
    
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                text = axes[1].text(j, i, f'{win_matrix_ece[i, j]:.0f}%',
                                   ha="center", va="center",
                                   color="white" if win_matrix_ece[i, j] > 50 else "black",
                                   fontsize=9, fontweight='bold')
    
    axes[1].set_title('(B) Calibration (ECE) Win Rate Matrix\n(Row better calibrated than Column)', 
                     fontweight='bold', pad=10)
    axes[1].set_xlabel('Model (Column)', fontweight='bold')
    axes[1].set_ylabel('Model (Row)', fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Win Rate (%)', fontweight='bold')
    
    plt.suptitle('Head-to-Head Model Comparisons', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(f'{save_dir}/fig3_winrate_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/fig3_winrate_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Figure 3: Win-Rate Matrix")


# ============================================================================
# FIGURE 4: CALIBRATION QUALITY ANALYSIS
# ============================================================================

def create_figure4_calibration_analysis(df, save_dir='figures'):
    """
    Multi-panel calibration analysis:
    - Confidence vs Accuracy
    - ECE breakdown by dataset
    - Brier score analysis
    - Overconfidence analysis
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Confidence vs Accuracy scatter
    ax1 = fig.add_subplot(gs[0, 0])
    
    for model in sorted(df['model'].unique()):
        data = df[df['model'] == model]
        ax1.scatter(data['confidence'], data['accuracy'],
                   label=MODEL_NAMES.get(model, model),
                   color=COLORS.get(model, 'gray'),
                   s=80, alpha=0.6, edgecolors='black', linewidth=0.8)
    
    # Perfect calibration line
    ax1.plot([0.5, 1.0], [0.5, 1.0], 'k--', linewidth=2, alpha=0.5, label='Perfect Calibration')
    
    ax1.set_xlabel('Average Confidence', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('(A) Confidence vs Accuracy', fontweight='bold', pad=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: ECE by dataset and model (grouped bar chart)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Focus on 5% labels for clarity
    low_label = df[df['label_fraction'] == 0.05]
    datasets = sorted(low_label['dataset'].unique())
    models_subset = ['Supervised_MLP', 'SSL_MLP', 'CalibSSL', 'XGBoost_Calibrated']
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, model in enumerate(models_subset):
        ece_values = [low_label[(low_label['dataset'] == ds) & (low_label['model'] == model)]['ece'].values[0] 
                      for ds in datasets]
        ax2.bar(x + i*width, ece_values, width,
               label=MODEL_NAMES.get(model, model),
               color=COLORS.get(model, 'gray'),
               edgecolor='black', linewidth=0.8)
    
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='ECE=0.05')
    ax2.set_xlabel('Dataset', fontweight='bold')
    ax2.set_ylabel('ECE', fontweight='bold')
    ax2.set_title('(B) ECE by Dataset (5% Labels)', fontweight='bold', pad=10)
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([ds.capitalize() for ds in datasets])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel C: Brier Score vs ECE
    ax3 = fig.add_subplot(gs[1, 0])
    
    for model in sorted(df['model'].unique()):
        data = df[df['model'] == model]
        ax3.scatter(data['ece'], data['brier'],
                   label=MODEL_NAMES.get(model, model),
                   color=COLORS.get(model, 'gray'),
                   s=100, alpha=0.6, edgecolors='black', linewidth=1)
    
    ax3.set_xlabel('ECE (Expected Calibration Error)', fontweight='bold')
    ax3.set_ylabel('Brier Score', fontweight='bold')
    ax3.set_title('(C) ECE vs Brier Score Correlation', fontweight='bold', pad=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Overconfidence analysis (confidence - accuracy)
    ax4 = fig.add_subplot(gs[1, 1])
    
    df['overconfidence'] = df['confidence'] - df['accuracy']
    
    models_to_plot = sorted(df['model'].unique())
    overconf_data = [df[df['model'] == m]['overconfidence'].values for m in models_to_plot]
    
    bp = ax4.boxplot(overconf_data, labels=[MODEL_NAMES.get(m, m) for m in models_to_plot],
                     patch_artist=True, showmeans=True)
    
    # Color boxes
    for patch, model in zip(bp['boxes'], models_to_plot):
        patch.set_facecolor(COLORS.get(model, 'gray'))
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
    
    ax4.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect Calibration')
    ax4.set_ylabel('Overconfidence (Confidence - Accuracy)', fontweight='bold')
    ax4.set_title('(D) Overconfidence Distribution', fontweight='bold', pad=10)
    ax4.set_xticklabels([MODEL_NAMES.get(m, m) for m in models_to_plot], rotation=45, ha='right')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Calibration Quality Analysis', fontsize=16, fontweight='bold')
    
    plt.savefig(f'{save_dir}/fig4_calibration_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/fig4_calibration_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Figure 4: Calibration Analysis")


# ============================================================================
# FIGURE 5: ABLATION STUDY & COMPONENT ANALYSIS
# ============================================================================

def create_figure5_ablation_study(df, save_dir='figures'):
    """
    Detailed ablation study showing contribution of each component
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Focus on neural network variants
    neural_models = ['Supervised_MLP', 'SSL_MLP', 'MLP_Calib_Only', 'CalibSSL']
    
    # Panel A: Component contribution (bar chart with error bars)
    ax = axes[0]
    
    metrics = ['accuracy', 'ece', 'brier']
    metric_names = ['Accuracy ↑', 'ECE ↓', 'Brier ↓']
    
    # Calculate means and stds
    model_stats = []
    for model in neural_models:
        stats = df[df['model'] == model][metrics].agg(['mean', 'std'])
        model_stats.append({
            'model': MODEL_NAMES.get(model, model),
            'accuracy_mean': stats['accuracy']['mean'],
            'accuracy_std': stats['accuracy']['std'],
            'ece_mean': stats['ece']['mean'],
            'ece_std': stats['ece']['std'],
            'brier_mean': stats['brier']['mean'],
            'brier_std': stats['brier']['std']
        })
    
    stats_df = pd.DataFrame(model_stats)
    
    x = np.arange(len(neural_models))
    width = 0.25
    
    # Normalize for visualization (so all metrics visible)
    acc_norm = stats_df['accuracy_mean'] / stats_df['accuracy_mean'].max()
    ece_norm = 1 - (stats_df['ece_mean'] / stats_df['ece_mean'].max())  # Invert since lower is better
    brier_norm = 1 - (stats_df['brier_mean'] / stats_df['brier_mean'].max())
    
    ax.bar(x - width, acc_norm, width, label='Accuracy', color='#2E7D32', edgecolor='black')
    ax.bar(x, ece_norm, width, label='Calibration (1-ECE)', color='#1976D2', edgecolor='black')
    ax.bar(x + width, brier_norm, width, label='Brier (inv.)', color='#C62828', edgecolor='black')
    
    ax.set_ylabel('Normalized Performance', fontweight='bold')
    ax.set_title('(A) Component Contribution (Normalized)', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES.get(m, m) for m in neural_models], rotation=20, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel B: Incremental improvements
    ax = axes[1]
    
    # Show improvement over baseline
    baseline = df[df['model'] == 'Supervised_MLP']
    baseline_acc = baseline['accuracy'].mean()
    baseline_ece = baseline['ece'].mean()
    
    improvements_acc = []
    improvements_ece = []
    
    for model in neural_models[1:]:  # Skip baseline
        model_data = df[df['model'] == model]
        acc_imp = (model_data['accuracy'].mean() - baseline_acc) * 100
        ece_imp = (baseline_ece - model_data['ece'].mean()) / baseline_ece * 100  # % improvement
        
        improvements_acc.append(acc_imp)
        improvements_ece.append(ece_imp)
    
    x = np.arange(len(neural_models[1:]))
    width = 0.35
    
    ax.bar(x - width/2, improvements_acc, width, label='Accuracy Δ (%)', 
           color='#4CAF50', edgecolor='black')
    ax.bar(x + width/2, improvements_ece, width, label='ECE Improvement (%)',
           color='#2196F3', edgecolor='black')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Improvement over Supervised (%)', fontweight='bold')
    ax.set_title('(B) Incremental Improvements', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES.get(m, m) for m in neural_models[1:]], rotation=20, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel C: Performance by label fraction (neural models only)
    ax = axes[2]
    
    for model in neural_models:
        data = df[df['model'] == model].groupby('label_fraction')['accuracy'].mean().reset_index()
        ax.plot(data['label_fraction'] * 100, data['accuracy'],
               marker='o', label=MODEL_NAMES.get(model, model),
               color=COLORS.get(model, 'gray'), linewidth=2.5, markersize=8)
    
    ax.set_xlabel('Label Fraction (%)', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('(C) Neural Network Variants - Accuracy', fontweight='bold', pad=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel D: Calibration by label fraction
    ax = axes[3]
    
    for model in neural_models:
        data = df[df['model'] == model].groupby('label_fraction')['ece'].mean().reset_index()
        ax.plot(data['label_fraction'] * 100, data['ece'],
               marker='o', label=MODEL_NAMES.get(model, model),
               color=COLORS.get(model, 'gray'), linewidth=2.5, markersize=8)
    
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='ECE=0.05')
    ax.set_xlabel('Label Fraction (%)', fontweight='bold')
    ax.set_ylabel('ECE', fontweight='bold')
    ax.set_title('(D) Neural Network Variants - Calibration', fontweight='bold', pad=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Ablation Study: Component Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(f'{save_dir}/fig5_ablation_study.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/fig5_ablation_study.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Figure 5: Ablation Study")


# ============================================================================
# FIGURE 6: STATISTICAL SIGNIFICANCE VISUALIZATION
# ============================================================================

def create_figure6_statistical_significance(df, save_dir='figures'):
    """
    Visualize statistical significance of CalibSSL vs baselines
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    from scipy.stats import ttest_rel
    from matplotlib.colors import TwoSlopeNorm  # For centered colormap
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    baselines = ['Supervised_MLP', 'SSL_MLP', 'XGBoost', 'XGBoost_Calibrated']
    
    # Panel A: P-value heatmap
    ax = axes[0]
    
    metrics = ['accuracy', 'ece']
    p_values = []
    
    for metric in metrics:
        calibssl_values = df[df['model'] == 'CalibSSL'].sort_values(['dataset', 'label_fraction'])[metric].values
        
        metric_pvals = []
        for baseline in baselines:
            baseline_values = df[df['model'] == baseline].sort_values(['dataset', 'label_fraction'])[metric].values
            
            if len(calibssl_values) != len(baseline_values):
                metric_pvals.append(1.0)  # No significance if different lengths
                continue
            
            if metric == 'accuracy':
                t_stat, p_val = ttest_rel(calibssl_values, baseline_values)
            else:  # ECE - lower is better
                t_stat, p_val = ttest_rel(baseline_values, calibssl_values)
            
            metric_pvals.append(p_val)
        
        p_values.append(metric_pvals)
    
    p_value_matrix = np.array(p_values)
    
    # Plot with significance annotations
    im = ax.imshow(p_value_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)
    
    ax.set_xticks(np.arange(len(baselines)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([MODEL_NAMES.get(b, b) for b in baselines], rotation=30, ha='right')
    ax.set_yticklabels(['Accuracy', 'ECE'])
    
    # Add significance stars
    for i in range(len(metrics)):
        for j in range(len(baselines)):
            p = p_value_matrix[i, j]
            if p < 0.001:
                sig_text = '***'
            elif p < 0.01:
                sig_text = '**'
            elif p < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            
            ax.text(j, i, f'{p:.4f}\n{sig_text}',
                   ha="center", va="center",
                   color="white" if p < 0.05 else "black",
                   fontsize=10, fontweight='bold')
    
    ax.set_title('(A) Statistical Significance (p-values)\nCalibSSL vs Baselines', 
                fontweight='bold', pad=10)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('p-value', fontweight='bold')
    
    # Panel B: Effect sizes (Cohen's d)
    ax = axes[1]
    
    effect_sizes = []
    
    for metric in metrics:
        calibssl_values = df[df['model'] == 'CalibSSL'].sort_values(['dataset', 'label_fraction'])[metric].values
        
        metric_effects = []
        for baseline in baselines:
            baseline_values = df[df['model'] == baseline].sort_values(['dataset', 'label_fraction'])[metric].values
            
            if len(calibssl_values) != len(baseline_values):
                metric_effects.append(0.0)
                continue
            
            diff = calibssl_values - baseline_values
            if metric == 'ece':
                diff = -diff  # Flip for ECE
            
            cohen_d = diff.mean() / (diff.std() + 1e-8)  # Add small constant to avoid division by zero
            metric_effects.append(cohen_d)
        
        effect_sizes.append(metric_effects)
    
    effect_matrix = np.array(effect_sizes)
    
    # Use TwoSlopeNorm for centered colormap
    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im2 = ax.imshow(effect_matrix, cmap='RdYlGn', norm=divnorm)
    
    ax.set_xticks(np.arange(len(baselines)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([MODEL_NAMES.get(b, b) for b in baselines], rotation=30, ha='right')
    ax.set_yticklabels(['Accuracy', 'ECE'])
    
    # Add effect size values
    for i in range(len(metrics)):
        for j in range(len(baselines)):
            d = effect_matrix[i, j]
            ax.text(j, i, f'{d:.2f}',
                   ha="center", va="center",
                   color="white" if abs(d) > 0.5 else "black",
                   fontsize=10, fontweight='bold')
    
    ax.set_title("(B) Effect Sizes (Cohen's d)\nCalibSSL vs Baselines",
                fontweight='bold', pad=10)
    
    cbar2 = plt.colorbar(im2, ax=ax)
    cbar2.set_label("Cohen's d", fontweight='bold')
    
    plt.suptitle('Statistical Analysis of CalibSSL Performance',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(f'{save_dir}/fig6_statistical_significance.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/fig6_statistical_significance.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Figure 6: Statistical Significance")

# ============================================================================
# SUPPLEMENTARY FIGURES
# ============================================================================

def create_supplementary_figures(df, save_dir='figures/supplementary'):
    """
    Create additional supplementary figures
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Supp Fig 1: All metrics correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    
    metrics = ['accuracy', 'f1', 'auc', 'ece', 'mce', 'brier', 'confidence']
    corr_matrix = df[metrics].corr()
    
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_yticklabels(metrics)
    
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center",
                          color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                          fontsize=9)
    
    ax.set_title('Metric Correlation Matrix', fontweight='bold', fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pearson Correlation', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/suppfig1_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Supplementary Figure 1: Correlation Matrix")
    
    # Supp Fig 2: Performance variance across datasets
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(['accuracy', 'ece', 'brier']):
        variance_data = df.groupby('model')[metric].std().sort_values(ascending=False)
        
        axes[idx].barh(range(len(variance_data)), variance_data.values,
                      color=[COLORS.get(m, 'gray') for m in variance_data.index],
                      edgecolor='black', linewidth=1.2)
        
        axes[idx].set_yticks(range(len(variance_data)))
        axes[idx].set_yticklabels([MODEL_NAMES.get(m, m) for m in variance_data.index])
        axes[idx].set_xlabel(f'{metric.upper()} Standard Deviation', fontweight='bold')
        axes[idx].set_title(f'{metric.capitalize()} Variance', fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Performance Stability Across Experiments', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/suppfig2_variance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Supplementary Figure 2: Variance Analysis")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_all_publication_figures(results_path='results/results.csv'):
    """
    Generate all publication-quality figures
    """
    print("="*70)
    print("PUBLICATION-GRADE VISUALIZATION SUITE".center(70))
    print("="*70)
    print()
    
    df = load_results(results_path)
    print()
    
    print("Generating main figures...")
    create_figure1_comprehensive_overview(df)
    create_figure2_per_dataset_analysis(df)
    create_figure3_winrate_matrix(df)
    create_figure4_calibration_analysis(df)
    create_figure5_ablation_study(df)
    create_figure6_statistical_significance(df)
    
    print("\nGenerating supplementary figures...")
    create_supplementary_figures(df)
    
    print("\n" + "="*70)
    print("✅ ALL PUBLICATION FIGURES GENERATED!".center(70))
    print("="*70)
    print("\nMain Figures (figures/):")
    print("  - fig1_comprehensive_overview.png/pdf")
    print("  - fig2_per_dataset_analysis.png/pdf")
    print("  - fig3_winrate_matrix.png/pdf")
    print("  - fig4_calibration_analysis.png/pdf")
    print("  - fig5_ablation_study.png/pdf")
    print("  - fig6_statistical_significance.png/pdf")
    print("\nSupplementary Figures (figures/supplementary/):")
    print("  - suppfig1_correlation_matrix.png")
    print("  - suppfig2_variance_analysis.png")
    print("="*70)


if __name__ == "__main__":
    generate_all_publication_figures()