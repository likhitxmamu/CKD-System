import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def define_and_train_models(X, y):
    """Define and train multiple models"""
    models = {
        'Logistic_Regression': LogisticRegression(),
        'SVM': SVC(probability=True),
        'Random_Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier(),
        'CatBoost': CatBoostClassifier(verbose=False)
    }
    
    # Train each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X, y)
    
    return models

def save_models(models, models_dir):
    """Save trained models"""
    os.makedirs(models_dir, exist_ok=True)
    for name, model in models.items():
        path = os.path.join(models_dir, f"{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(model, f)


