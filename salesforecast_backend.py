from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import base64
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store model and data
model = None
feature_columns = None
df_global = None
model_metrics = {}

def load_and_preprocess(file_path):
    """Load and preprocess the sales data"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Handle different possible column names
        column_mapping = {}
        for col in df.columns:
            if 'fuel' in col and 'price' in col:
                column_mapping[col] = 'fuel_price'
            elif 'weekly' in col and 'sales' in col:
                column_mapping[col] = 'weekly_sales'
            elif 'holiday' in col:
                column_mapping[col] = 'holiday_flag'
            elif col in ['temp', 'temperature']:
                column_mapping[col] = 'temperature'
            elif col == 'cpi':
                column_mapping[col] = 'cpi'
            elif 'unemployment' in col:
                column_mapping[col] = 'unemployment'
            elif col == 'store':
                column_mapping[col] = 'store'
            elif col == 'date':
                column_mapping[col] = 'date'
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Convert date column to datetime
        if 'date' in df.columns:
            date_formats = ['%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']
            for fmt in date_formats:
                try:
                    df['date'] = pd.to_datetime(df['date'], format=fmt, errors='raise')
                    break
                except:
                    continue
            else:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'store':
                df[col] = df[col].fillna(df[col].median())
        
        # Create default values for missing columns
        if 'store' not in df.columns:
            df['store'] = 1
        if 'holiday_flag' not in df.columns:
            df['holiday_flag'] = 0
        if 'temperature' not in df.columns:
            df['temperature'] = 70.0
        if 'fuel_price' not in df.columns:
            df['fuel_price'] = 3.5
        if 'cpi' not in df.columns:
            df['cpi'] = 200.0
        if 'unemployment' not in df.columns:
            df['unemployment'] = 7.0
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def create_time_features(df):
    """Create time-based features"""
    df = df.copy()
    df = df.sort_values(['store', 'date'])
    
    # Basic time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Cyclical encoding for seasonality
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lag features
    for store in df['store'].unique():
        store_mask = df['store'] == store
        store_data = df[store_mask].sort_values('date')
        
        for lag in [1, 2, 4, 8]:
            df.loc[store_mask, f'sales_lag_{lag}'] = store_data['weekly_sales'].shift(lag)
        
        for window in [4, 8, 12]:
            df.loc[store_mask, f'sales_roll_mean_{window}'] = store_data['weekly_sales'].rolling(window=window, min_periods=1).mean()
            df.loc[store_mask, f'sales_roll_std_{window}'] = store_data['weekly_sales'].rolling(window=window, min_periods=1).std()
    
    # Fill NaN values
    lag_cols = [col for col in df.columns if 'lag_' in col or 'roll_' in col]
    for col in lag_cols:
        df[col] = df[col].fillna(df[col].median())
    
    return df

def create_features(df):
    """Create feature matrix for modeling"""
    df_features = create_time_features(df)
    
    feature_cols = [
        'holiday_flag', 'temperature', 'fuel_price', 'cpi', 'unemployment',
        'week_sin', 'week_cos', 'month_sin', 'month_cos',
        'month', 'quarter'
    ]
    
    lag_cols = [col for col in df_features.columns if 'lag_' in col or 'roll_' in col]
    feature_cols.extend(lag_cols)
    
    available_cols = [col for col in feature_cols if col in df_features.columns]
    
    X = df_features[available_cols]
    y = df_features['weekly_sales']
    
    return X, y, df_features, available_cols

def train_model(df):
    """Train the XGBoost model"""
    global model, feature_columns, model_metrics
    
    X, y, df_features, available_cols = create_features(df)
    feature_columns = available_cols
    
    X = X.fillna(X.median())
    df_features = df_features.sort_values('date')
    X = X.loc[df_features.index]
    y = y.loc[df_features.index]
    
    # Time series split
    split_idx = int(len(df_features) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    # Train model
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mape = mean_absolute_percentage_error(y_train, train_pred) * 100
    test_mape = mean_absolute_percentage_error(y_test, test_pred) * 100
    
    model_metrics = {
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_mape': float(train_mape),
        'test_mape': float(test_mape),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features_used': len(available_cols)
    }
    
    return model, model_metrics

# API Routes

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'message': 'Sales Forecasting API is running',
        'status': 'healthy',
        'model_trained': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload', methods=['POST'])
def upload_data():
    """Upload and process CSV data"""
    global df_global, model, model_metrics
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        temp_path = 'temp_data.csv'
        file.save(temp_path)
        
        # Load and preprocess data
        df_global = load_and_preprocess(temp_path)
        
        if df_global is None:
            return jsonify({'error': 'Failed to process data file'}), 400
        
        # Train model
        model, metrics = train_model(df_global)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return jsonify({
            'message': 'Data uploaded and model trained successfully',
            'data_info': {
                'total_records': len(df_global),
                'stores': int(df_global['store'].nunique()),
                'date_range': {
                    'start': df_global['date'].min().isoformat(),
                    'end': df_global['date'].max().isoformat()
                }
            },
            'model_performance': metrics
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions for given input data"""
    global model, feature_columns
    
    if model is None:
        return jsonify({'error': 'Model not trained. Please upload data first.'}), 400
    
    try:
        data = request.json
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([data])
        
        # Add default values for missing columns
        for col in feature_columns:
            if col not in input_df.columns:
                if col == 'holiday_flag':
                    input_df[col] = 0
                elif col == 'temperature':
                    input_df[col] = 70.0
                elif col == 'fuel_price':
                    input_df[col] = 3.5
                elif col == 'cpi':
                    input_df[col] = 200.0
                elif col == 'unemployment':
                    input_df[col] = 7.0
                else:
                    input_df[col] = 0.0
        
        # Ensure all feature columns are present
        input_df = input_df[feature_columns]
        input_df = input_df.fillna(0)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'input_features': data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Make predictions for multiple input records"""
    global model, feature_columns
    
    if model is None:
        return jsonify({'error': 'Model not trained. Please upload data first.'}), 400
    
    try:
        data = request.json
        
        if not isinstance(data, list):
            return jsonify({'error': 'Input should be a list of records'}), 400
        
        # Create DataFrame from input data
        input_df = pd.DataFrame(data)
        
        # Add default values for missing columns
        for col in feature_columns:
            if col not in input_df.columns:
                if col == 'holiday_flag':
                    input_df[col] = 0
                elif col == 'temperature':
                    input_df[col] = 70.0
                elif col == 'fuel_price':
                    input_df[col] = 3.5
                elif col == 'cpi':
                    input_df[col] = 200.0
                elif col == 'unemployment':
                    input_df[col] = 7.0
                else:
                    input_df[col] = 0.0
        
        # Ensure all feature columns are present
        input_df = input_df[feature_columns]
        input_df = input_df.fillna(0)
        
        # Make predictions
        predictions = model.predict(input_df)
        
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'index': i,
                'prediction': float(pred),
                'input': data[i]
            })
        
        return jsonify({
            'predictions': results,
            'total_predictions': len(predictions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction error: {str(e)}'}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the trained model"""
    global model, feature_columns, model_metrics, df_global
    
    if model is None:
        return jsonify({'error': 'Model not trained. Please upload data first.'}), 400
    
    # Feature importance
    feature_importance = []
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for i, col in enumerate(feature_columns):
            feature_importance.append({
                'feature': col,
                'importance': float(importances[i])
            })
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    return jsonify({
        'model_type': 'XGBoost Regressor',
        'features_count': len(feature_columns),
        'feature_names': feature_columns,
        'performance_metrics': model_metrics,
        'feature_importance': feature_importance[:10],  # Top 10 features
        'data_info': {
            'total_records': len(df_global) if df_global is not None else 0,
            'stores': int(df_global['store'].nunique()) if df_global is not None else 0
        }
    })

@app.route('/forecast', methods=['POST'])
def forecast():
    """Generate forecast for future periods"""
    global model, feature_columns, df_global
    
    if model is None:
        return jsonify({'error': 'Model not trained. Please upload data first.'}), 400
    
    try:
        data = request.json
        store_id = data.get('store_id', 1)
        periods = data.get('periods', 4)  # Number of weeks to forecast
        
        if df_global is None:
            return jsonify({'error': 'No historical data available'}), 400
        
        # Get last known data for the store
        store_data = df_global[df_global['store'] == store_id].sort_values('date')
        if len(store_data) == 0:
            return jsonify({'error': f'No data found for store {store_id}'}), 400
        
        last_record = store_data.iloc[-1]
        last_date = last_record['date']
        
        forecasts = []
        
        for i in range(1, periods + 1):
            # Create future date
            future_date = last_date + timedelta(weeks=i)
            
            # Create input features for prediction
            input_data = {
                'holiday_flag': data.get('holiday_flag', 0),
                'temperature': data.get('temperature', last_record['temperature']),
                'fuel_price': data.get('fuel_price', last_record['fuel_price']),
                'cpi': data.get('cpi', last_record['cpi']),
                'unemployment': data.get('unemployment', last_record['unemployment']),
                'week_sin': np.sin(2 * np.pi * future_date.isocalendar().week / 52),
                'week_cos': np.cos(2 * np.pi * future_date.isocalendar().week / 52),
                'month_sin': np.sin(2 * np.pi * future_date.month / 12),
                'month_cos': np.cos(2 * np.pi * future_date.month / 12),
                'month': future_date.month,
                'quarter': (future_date.month - 1) // 3 + 1
            }
            
            # Add lag features (use last known values)
            for col in feature_columns:
                if col.startswith('sales_lag_') or col.startswith('sales_roll_'):
                    input_data[col] = last_record.get(col, 0)
            
            # Ensure all features are present
            input_df = pd.DataFrame([input_data])
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[feature_columns]
            prediction = model.predict(input_df)[0]
            
            forecasts.append({
                'date': future_date.isoformat(),
                'predicted_sales': float(prediction),
                'week': i
            })
        
        return jsonify({
            'store_id': store_id,
            'forecast_periods': periods,
            'forecasts': forecasts,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Forecast error: {str(e)}'}), 500

@app.route('/stores', methods=['GET'])
def get_stores():
    """Get list of available stores"""
    global df_global
    
    if df_global is None:
        return jsonify({'error': 'No data available. Please upload data first.'}), 400
    
    stores = df_global['store'].unique().tolist()
    store_info = []
    
    for store in stores:
        store_data = df_global[df_global['store'] == store]
        store_info.append({
            'store_id': int(store),
            'total_records': len(store_data),
            'date_range': {
                'start': store_data['date'].min().isoformat(),
                'end': store_data['date'].max().isoformat()
            },
            'avg_weekly_sales': float(store_data['weekly_sales'].mean()),
            'total_sales': float(store_data['weekly_sales'].sum())
        })
    
    return jsonify({
        'total_stores': len(stores),
        'stores': store_info
    })

if __name__ == '__main__':
    print("Starting Sales Forecasting API...")
    print("API Endpoints:")
    print("- GET  /                 : Health check")
    print("- POST /upload           : Upload CSV data and train model")
    print("- POST /predict          : Single prediction")
    print("- POST /batch_predict    : Batch predictions")
    print("- GET  /model_info       : Model information")
    print("- POST /forecast         : Generate forecasts")
    print("- GET  /stores           : Get store information")
    
    app.run(debug=True, host='0.0.0.0', port=5000)