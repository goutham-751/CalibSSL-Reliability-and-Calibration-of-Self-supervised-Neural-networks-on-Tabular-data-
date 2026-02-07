# Self-Supervised Neural Network Calibration on Tabular Data

## Research Environment

This project focuses on calibration of self-supervised neural networks for tabular data.

## Setup Instructions

### 1. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.\venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import torch; import sklearn; import netcal; print('Setup successful!')"
```

## Project Structure

```
Research paper/
├── venv/                    # Virtual environment
├── Datasets/               # Your datasets (CSV files)
│   ├── adult_income/       # Adult Income dataset
│   ├── wine+quality/       # Wine Quality (red & white)
│   ├── bank+marketing/     # Bank Marketing dataset
│   └── archive (3)/        # Credit Card Default dataset
├── notebooks/              # Jupyter notebooks for experiments
│   ├── *.ipynb            # Main analysis notebooks
│   ├── data_loading_*.py  # CSV loading templates
│   ├── COLAB_SETUP.md     # Google Colab instructions
│   └── README_CSV_LOADING.md  # CSV loading guide
├── src/                    # Source code
│   ├── models/            # Neural network models
│   ├── calibration/       # Calibration methods
│   ├── data/              # Data processing utilities
│   │   └── data_loaders.py  # CSV data loaders
│   └── utils/             # Helper functions
├── experiments/           # Experiment results
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Loading Your Datasets

All datasets are stored as CSV files in the `Datasets/` folder. You have two options:

### Option 1: Use the Unified Data Loader

```python
from src.data.data_loaders import load_adult_income, load_wine_quality

# Load Adult Income dataset
df = load_adult_income('../Datasets/adult_income/adultincome/adult.csv')

# Load Wine Quality dataset (combines red & white)
df = load_wine_quality(
    '../Datasets/wine+quality/winequality-red.csv',
    '../Datasets/wine+quality/winequality-white.csv'
)
```

### Option 2: Use Individual Templates

Each dataset has a dedicated loading template in `notebooks/`:
- `data_loading_adult_income.py`
- `data_loading_wine_quality.py`
- `data_loading_bank_marketing.py`
- `data_loading_credit_default.py`

### For Google Colab

See **`notebooks/COLAB_SETUP.md`** for detailed instructions on uploading and using CSV files in Colab.

**Quick Summary:**
```bash
# Test all datasets load correctly
python src/data/data_loaders.py
```

## Key Libraries Included

- **Deep Learning:** PyTorch, TensorFlow, PyTorch Lightning
- **Calibration:** netcal, uncertainty-calibration
- **Tabular Data:** pandas, scikit-learn, category-encoders
- **Visualization:** matplotlib, seaborn, plotly
- **Hyperparameter Tuning:** Optuna
- **Baselines:** XGBoost, LightGBM

## Research Topics

1. Self-supervised learning on tabular data
2. Model calibration techniques
3. Uncertainty quantification
4. Performance evaluation

## Getting Started

After activating the environment, start with:
```bash
jupyter notebook
```

This will open Jupyter Notebook where you can begin your experiments.
