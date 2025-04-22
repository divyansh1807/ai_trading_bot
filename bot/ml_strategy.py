import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

class MLTradingStrategy:
    def __init__(self, model_type='random_forest', load_existing=True):
        """
        Initialize the ML trading strategy
        
        Args:
            model_type (str): Type of ML model to use ('random_forest' or 'gradient_boosting')
            load_existing (bool): Whether to load an existing trained model if available
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Try to load existing model if requested
        if load_existing:
            model_path = os.path.join('models', f'{model_type}_model.joblib')
            scaler_path = os.path.join('models', f'{model_type}_scaler.joblib')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.model = joblib.load(model_path)
                    self.scaler = joblib.load(scaler_path)
                    print(f"✅ Loaded existing {model_type} model")
                except Exception as e:
                    print(f"❌ Failed to load existing model: {str(e)}")
                    self.model = None
    
    def prepare_features(self, df):
        """
        Prepare features for the ML model
        
        Args:
            df (DataFrame): Price data with indicators
            
        Returns:
            DataFrame: Feature DataFrame
        """
        # Create a copy to avoid modifying the original
        data = df.copy()
        
        # Technical indicators (already calculated in the AdvancedTradingBot)
        # We'll use them as features
        
        # Price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Volatility features
        data['volatility_5'] = data['returns'].rolling(window=5).std()
        data['volatility_15'] = data['returns'].rolling(window=15).std()
        
        # Volume features
        data['volume_change'] = data['volume'].pct_change()
        data['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()
        
        # Price distance from moving averages
        data['price_sma_ratio_short'] = data['close'] / data['SMA_short']
        data['price_sma_ratio_long'] = data['close'] / data['SMA_long']
        
        # RSI momentum
        data['rsi_change'] = data['RSI'].diff()
        
        # MACD features
        data['macd_diff'] = data['MACD'] - data['Signal_Line']
        data['macd_diff_change'] = data['macd_diff'].diff()
        
        # Bollinger Band features
        data['bb_width'] = (data['Upper_Band'] - data['Lower_Band']) / data['20MA']
        data['bb_position'] = (data['close'] - data['Lower_Band']) / (data['Upper_Band'] - data['Lower_Band'])
        
        # Target variable: price direction (1 if price goes up in next period, 0 otherwise)
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
        
        # Drop NaN values
        data = data.dropna()
        
        # Select feature columns
        self.feature_columns = [
            'returns', 'log_returns', 
            'volatility_5', 'volatility_15',
            'volume_change', 'volume_ma_ratio',
            'price_sma_ratio_short', 'price_sma_ratio_long',
            'RSI', 'rsi_change',
            'MACD', 'Signal_Line', 'macd_diff', 'macd_diff_change',
            'bb_width', 'bb_position'
        ]
        
        return data
    
    def train(self, df, test_size=0.2, optimize=False):
        """
        Train the ML model
        
        Args:
            df (DataFrame): Price data with indicators
            test_size (float): Proportion of data to use for testing
            optimize (bool): Whether to optimize hyperparameters
            
        Returns:
            dict: Training metrics
        """
        # Prepare features
        data = self.prepare_features(df)
        
        # Split features and target
        X = data[self.feature_columns]
        y = data['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, shuffle=False)
        
        # Initialize model based on type
        if self.model_type == 'random_forest':
            if optimize:
                # Grid search for hyperparameter optimization
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                grid_search = GridSearchCV(
                    RandomForestClassifier(random_state=42),
                    param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                self.model = RandomForestClassifier(random_state=42)
                self.model.fit(X_train, y_train)
        
        elif self.model_type == 'gradient_boosting':
            if optimize:
                # Grid search for hyperparameter optimization
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                grid_search = GridSearchCV(
                    GradientBoostingClassifier(random_state=42),
                    param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                self.model = GradientBoostingClassifier(random_state=42)
                self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        print(f"Model performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        # Save model and scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, os.path.join('models', f'{self.model_type}_model.joblib'))
        joblib.dump(self.scaler, os.path.join('models', f'{self.model_type}_scaler.joblib'))
        
        return metrics
    
    def predict(self, df):
        """
        Make predictions using the trained model
        
        Args:
            df (DataFrame): Price data with indicators
            
        Returns:
            Series: Predicted probabilities of price going up
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() first.")
        
        # Prepare features
        data = self.prepare_features(df)
        
        # Select feature columns
        X = data[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions (probability of price going up)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Add predictions to the DataFrame
        data['ml_prediction'] = probabilities
        
        # Generate signals based on probability threshold
        data['ml_signal'] = 0
        data.loc[data['ml_prediction'] > 0.6, 'ml_signal'] = 1  # Strong buy signal
        data.loc[data['ml_prediction'] < 0.4, 'ml_signal'] = -1  # Strong sell signal
        
        return data[['ml_prediction', 'ml_signal']]
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        
        Returns:
            DataFrame: Feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() first.")
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            
            # Create DataFrame
            feature_importance = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': importance
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            return feature_importance
        else:
            return None
