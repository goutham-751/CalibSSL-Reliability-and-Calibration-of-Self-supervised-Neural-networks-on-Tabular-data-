import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch

class TabularDataset:
    """Unified dataset loader for all 5 datasets"""
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess(self):
        """Load and preprocess specific dataset"""
        
        if self.dataset_name == 'adult':
            return self._load_adult()
        elif self.dataset_name == 'bank':
            return self._load_bank()
        elif self.dataset_name == 'credit':
            return self._load_credit()
        elif self.dataset_name == 'jannis':
            return self._load_jannis()
        elif self.dataset_name == 'diabetes':
            return self._load_diabetes()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_adult(self):
        """Adult Income dataset"""
        df = pd.read_csv('data/data/raw/adult.csv')
        
        # Target is 'class' column
        y = df['class']
        X = df.drop('class', axis=1)
        
        # Handle categorical variables
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
        # Encode target
        y = self.label_encoder.fit_transform(y)
        
        return X.values, y
    
    def _load_bank(self):
        """Bank Marketing dataset"""
        df = pd.read_csv('data/data/raw/bank.csv')
        
        # Target column (OpenML uses 'Class' with capital C)
        target_col = next(col for col in df.columns if col.lower() == 'class')
        y = df[target_col]
        X = df.drop(target_col, axis=1)
        
        # Handle categorical
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
        y = self.label_encoder.fit_transform(y)
        
        return X.values, y
    
    def _load_credit(self):
        """Credit Default dataset (German Credit from OpenML)"""
        df = pd.read_csv('data/data/raw/credit.csv')
        
        # Target is 'class' column (good/bad)
        y = df['class']
        X = df.drop('class', axis=1)
        
        # Handle categorical variables
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
        y = self.label_encoder.fit_transform(y)
        
        return X.values, y
    
    def _load_jannis(self):
        """Jannis dataset (83k rows, tabular benchmark)"""
        df = pd.read_csv('data/data/raw/jannis.csv')
        
        # Target column is 'class'
        y = self.label_encoder.fit_transform(df['class'])
        X = df.drop('class', axis=1)
        
        # Handle categorical
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
        return X.values.astype(np.float32), y
    
    def _load_diabetes(self):
        """Pima Indians Diabetes dataset (OpenML)"""
        df = pd.read_csv('data/data/raw/diabetes.csv')
        
        # Target is 'class' column (tested_positive / tested_negative)
        y = self.label_encoder.fit_transform(df['class'])
        X = df.drop('class', axis=1).values
        
        return X, y
    
    def create_splits(self, X, y, label_fraction=1.0, test_size=0.2, random_state=42):
        """
        Create train/val/test splits with limited labels
        
        Returns:
        - X_labeled, y_labeled: Labeled training data
        - X_unlabeled: Unlabeled data (for SSL pretraining)
        - X_test, y_test: Test data
        """
        
        # First split: train vs test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Create labeled subset
        if label_fraction < 1.0:
            n_labeled = int(len(X_train) * label_fraction)
            
            # Stratified sampling for labeled subset
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=label_fraction, random_state=random_state)
            labeled_idx, unlabeled_idx = next(sss.split(X_train, y_train))
            
            X_labeled = X_train[labeled_idx]
            y_labeled = y_train[labeled_idx]
            X_unlabeled = X_train  # Keep ALL for SSL
        else:
            X_labeled = X_train
            y_labeled = y_train
            X_unlabeled = X_train
        
        # Convert to torch tensors
        X_labeled = torch.FloatTensor(X_labeled)
        y_labeled = torch.LongTensor(y_labeled)
        X_unlabeled = torch.FloatTensor(X_unlabeled)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)
        
        return X_labeled, y_labeled, X_unlabeled, X_test, y_test

# ============ UTILITY FUNCTION ============
def load_dataset(dataset_name, label_fraction=1.0):
    """
    Main function to load any dataset
    
    Usage:
        X_lab, y_lab, X_unlab, X_test, y_test = load_dataset('adult', label_fraction=0.1)
    """
    loader = TabularDataset(dataset_name)
    X, y = loader.load_and_preprocess()
    return loader.create_splits(X, y, label_fraction=label_fraction)


# ============ TEST ALL DATASETS ============
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    datasets = ['adult', 'bank', 'credit', 'covertype', 'diabetes']
    
    print("Testing all datasets...\n")
    for ds in datasets:
        print(f"Loading {ds}...")
        try:
            X_lab, y_lab, X_unlab, X_test, y_test = load_dataset(ds, label_fraction=0.1)
            print(f"  [OK] {ds:12s} | Labeled: {len(X_lab):6d} | Unlabeled: {len(X_unlab):6d} | Test: {len(X_test):6d} | Features: {X_lab.shape[1]:3d} | Classes: {len(torch.unique(y_test))}")
        except Exception as e:
            print(f"  [FAIL] {ds:12s} | Error: {e}")
        print()