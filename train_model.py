"""
Model Training Script for Mumbai House Price Prediction
Trains multiple regression models and selects the best one
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_preprocessor import DataPreprocessor
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare the dataset"""
    print("Loading dataset...")
    df = pd.read_csv('mumbai-house-price-data-cleaned.csv')
    print(f"Dataset shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess the data for training"""
    print("\nPreprocessing data...")
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")
    
    # Select relevant features
    features = ['property_type', 'area', 'locality', 'bedroom_num', 'bathroom_num', 
                'balcony_num', 'furnished', 'age', 'total_floors', 
                'latitude', 'longitude']
    
    X = df[features].copy()
    y = df['price'].copy()
    
    # Remove outliers (prices less than 100k and more than 100Cr)
    X = X[(y >= 100000) & (y <= 100000000)].copy()
    y = y[(y >= 100000) & (y <= 100000000)]
    
    print(f"After removing outliers: {X.shape}")
    
    # Create preprocessor
    preprocessor = DataPreprocessor()
    
    # Encode categorical variables
    categorical_cols = ['property_type', 'locality', 'furnished']
    X_encoded = preprocessor.fit_transform(X, categorical_cols)
    
    return X_encoded, y, preprocessor, categorical_cols

def train_and_evaluate_models(X, y):
    """Train and evaluate multiple models"""
    print("\n" + "="*60)
    print("Training Multiple Models")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'-'*60}")
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
        
        print(f"Train R² Score: {train_r2:.4f}")
        print(f"Test R² Score: {test_r2:.4f}")
        print(f"Train MAE: Rs {train_mae/1e6:.2f} Million")
        print(f"Test MAE: Rs {test_mae/1e6:.2f} Million")
        print(f"Train RMSE: Rs {train_rmse/1e6:.2f} Million")
        print(f"Test RMSE: Rs {test_rmse/1e6:.2f} Million")
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['test_r2'])
    print(f"\n{'='*60}")
    print(f"Best Model: {best_model_name}")
    print(f"Test R² Score: {results[best_model_name]['test_r2']:.4f}")
    print(f"{'='*60}")
    
    return results, best_model_name

def hyperparameter_tuning(best_model, X_train, y_train):
    """Fine-tune the best model"""
    print("\n" + "="*60)
    print("Hyperparameter Tuning")
    print("="*60)
    
    if isinstance(best_model, RandomForestRegressor):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    elif isinstance(best_model, GradientBoostingRegressor):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
        base_model = GradientBoostingRegressor(random_state=42)
    
    else:
        print("Skipping hyperparameter tuning for this model type.")
        return best_model
    
    print(f"\nSearching for best hyperparameters...")
    print("This may take a few minutes...\n")
    
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=3, 
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def save_model(model, preprocessor):
    """Save the trained model and preprocessor"""
    print("\n" + "="*60)
    print("Saving Model")
    print("="*60)
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("[OK] Model saved as 'model.pkl'")
    
    # Save preprocessor
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("[OK] Encoder saved as 'encoder.pkl'")
    
    print("\n[OK] Model training complete! You can now run the Streamlit app.")

def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print(" " * 15 + "Mumbai House Price Prediction - Model Training")
    print("="*80 + "\n")
    
    # Load data
    df = load_data()
    
    # Preprocess data
    X, y, preprocessor, categorical_cols = preprocess_data(df)
    
    # Train and evaluate models
    results, best_model_name = train_and_evaluate_models(X, y)
    
    # Get best model
    best_model = results[best_model_name]['model']
    
    # Optional hyperparameter tuning (commented out for faster training)
    # You can uncomment this for better performance
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # best_model = hyperparameter_tuning(best_model, X_train, y_train)
    
    # Final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    final_r2 = r2_score(y_test, y_pred)
    final_mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"Final Model Performance")
    print(f"{'='*60}")
    print(f"Model: {best_model_name}")
    print(f"R² Score: {final_r2:.4f}")
    print(f"MAE: Rs {final_mae/1e6:.2f} Million")
    print(f"{'='*60}\n")
    
    # Save model
    save_model(best_model, preprocessor)
    
    print("\n" + "="*80)
    print(" " * 30 + "Training Complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

