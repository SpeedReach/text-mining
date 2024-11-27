import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import os
from itertools import product

def load_data(file_path):
    """Load and prepare data from Excel file"""
    df = pd.read_excel(file_path)
    return df

def create_run_file(results_df, filename):
    """Create a TREC-style run file from results DataFrame"""
    results_df = results_df.sort_values(['query_id', 'score'], ascending=[True, False])
    results_df['rank'] = results_df.groupby('query_id').cumcount() + 1
    
    os.makedirs('runs', exist_ok=True)
    with open(f'runs/{filename}.run', 'w') as f:
        for _, row in results_df.iterrows():
            f.write(f"{row['query_id']} Q0 {row['doc_id']} {row['rank']} {row['score']:.5f} Exp\n")

def train_and_evaluate(model, X_train_scaled, y_train, X_test_scaled, test_df, model_name):
    """Train model and create run file"""
    model.fit(X_train_scaled, y_train)
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:  # For models like SVM that don't have predict_proba
        y_pred_proba = model.decision_function(X_test_scaled)
    
    results_df = pd.DataFrame({
        'query_id': test_df['Query ID'],
        'doc_id': test_df['Doc ID'],
        'score': y_pred_proba
    })
    
    create_run_file(results_df, model_name)
    print(f"Results written to {model_name}.run")

def main():
    # Load data
    train_df = load_data('training_data.xlsx')
    test_df = load_data('test_data.xlsx')
    
    # Prepare features
    X_train = train_df[['Score 1', 'Score 2', 'Score 3']]
    y_train = train_df['Relevance']
    X_test = test_df[['Score 1', 'Score 2', 'Score 3']]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate positive class weight for XGBoost
    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    # Define models and their parameters
    models = {
        'xgboost': {
            'base': XGBClassifier,
            'params': {
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'n_estimators': [100, 200],
                'scale_pos_weight': [pos_weight],
                'random_state': [42]
            }
        },
        'random_forest': {
            'base': RandomForestClassifier,
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5],
                'random_state': [42]
            }
        },
        'gradient_boosting': {
            'base': GradientBoostingClassifier,
            'params': {
                'learning_rate': [0.01, 0.1],
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'random_state': [42]
            }
        },
        'svm': {
            'base': SVC,
            'params': {
                'kernel': ['rbf', 'linear'],
                'C': [1, 10],
                'probability': [True],
                'random_state': [42]
            }
        }
    }
    
    # Train and evaluate each model configuration
    for model_name, model_info in models.items():
        base_model = model_info['base']
        param_grid = model_info['params']
        
        # Generate all parameter combinations
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = [dict(zip(param_keys, v)) 
                            for v in product(*param_values)]
        
        # Train and evaluate each parameter combination
        for i, params in enumerate(param_combinations):
            # Create descriptive name
            param_str = '_'.join(f"{k[:2]}{v}" for k, v in params.items() 
                               if k not in ['random_state', 'probability', 'scale_pos_weight'])
            run_name = f"{model_name}_{param_str}"
            
            print(f"\nTraining {model_name} model {i+1}/{len(param_combinations)}:")
            print(f"Parameters: {params}")
            
            # Initialize and train model
            model = base_model(**params)
            train_and_evaluate(
                model,
                X_train_scaled,
                y_train,
                X_test_scaled,
                test_df,
                run_name
            )

if __name__ == "__main__":
    main()