import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.model import define_and_train_models, save_models
from src.preprocessing import preprocess_data

# Constants
DATA_PATH = os.path.join(project_root, 'data', 'raw.csv')
TARGET_COLUMN = 'classification'
MODELS_DIR = os.path.join(project_root, 'models')

def main():
    try:
        # Preprocessing
        X, y = preprocess_data(DATA_PATH, TARGET_COLUMN)
        
        # Train the models
        trained_models = define_and_train_models(X, y)
        
        # Create models directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Save the models
        save_models(trained_models, models_dir=MODELS_DIR)
        
    except FileNotFoundError:
        print(f"Error: Could not find data file at {DATA_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during model training/saving: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()