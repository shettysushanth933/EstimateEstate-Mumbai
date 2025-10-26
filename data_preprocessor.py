"""
Data Preprocessing utilities for Mumbai House Price Prediction
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    """Handle data preprocessing and encoding"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        
    def fit_transform(self, df, categorical_columns):
        """Fit and transform categorical columns"""
        df_processed = df.copy()
        
        for col in categorical_columns:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
                else:
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        return df_processed
    
    def transform(self, df):
        """Transform new data"""
        df_processed = df.copy()
        
        for col, encoder in self.label_encoders.items():
            if col in df_processed.columns:
                # Handle unseen labels by using the most common label
                try:
                    # Try to transform with the existing encoder
                    df_processed[col] = encoder.transform(df_processed[col])
                except ValueError:
                    # If we encounter a new label, map it to the first known label
                    df_processed[col] = df_processed[col].apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else 0
                    )
        
        return df_processed

