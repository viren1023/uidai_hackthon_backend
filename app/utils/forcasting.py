import pandas as pd
import numpy as np
from pymongo import MongoClient
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import joblib
import json
from datetime import timedelta, datetime
import os

# ============================ CONFIG ============================
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "your_db_name"

COLLECTIONS = {
    "biometric": {"collection_name": "biometric_collection", "targets": ["bio_0_5", "bio_5_17"]},
    "demographic": {"collection_name": "demographic_collection", "targets": ["demo_0_5", "demo_5_17"]},
    "enrollment": {"collection_name": "enrollment_collection", "targets": ["enroll_0_5", "enroll_5_17", "enroll_18_plus"]}
}

MODEL_DIR = "saved_models"
WEIGHTS_FILE = os.path.join(MODEL_DIR, "weights.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================ HELPER FUNCTIONS ============================

def load_collection_data(collection_name, state, district=None, collection_type=None):
    """
    Load and aggregate data from MongoDB for a given collection and state/district.
    Returns DataFrame with columns: date + target_cols.
    """
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[collection_name]

    query = {"state": state}
    if district:
        query["district"] = district

    cursor = collection.find(query).sort("date", 1)
    data = list(cursor)

    if not data:
        print(f"No data found for {state}, {district if district else 'ALL'} in {collection_name}")
        return None

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])

    # Select target metrics
    if collection_type == "biometric":
        target_cols = ["bio_0_5", "bio_5_17"]
    elif collection_type == "demographic":
        target_cols = ["demo_0_5", "demo_5_17"]
    elif collection_type == "enrollment":
        # Combine 5_17 + 18_plus into one column
        df['age_5_18_plus'] = df['mertics'].apply(lambda x: x.get('enroll_5_17',0)+x.get('enroll_18_plus',0) if isinstance(x, dict) else 0)
        df['enroll_0_5'] = df['mertics'].apply(lambda x: x.get('enroll_0_5',0) if isinstance(x, dict) else 0)
        target_cols = ['enroll_0_5','age_5_18_plus']
    else:
        raise ValueError("Invalid collection_type")

    if collection_type != "enrollment":
        # For demo/bio
        for col in target_cols:
            df[col] = df['mertics'].apply(lambda x: x.get(col, 0) if isinstance(x, dict) else 0)

    # Aggregate by date
    agg_dict = {col: 'sum' for col in target_cols}
    daily_df = df.groupby('date').agg(agg_dict).reset_index()

    # Fill missing dates
    date_range = pd.date_range(daily_df['date'].min(), daily_df['date'].max())
    daily_df = daily_df.set_index('date').reindex(date_range, fill_value=0).reset_index()
    daily_df.rename(columns={'index':'date'}, inplace=True)

    return daily_df, target_cols

def create_time_features(df, target_col):
    """
    Create lag features, rolling stats, trend, and EMA features
    """
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    # Lag features
    for lag in [1,7,14,30]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    # Rolling stats
    for window in [7,14,30]:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window, min_periods=1).std()
        df[f'rolling_min_{window}'] = df[target_col].rolling(window, min_periods=1).min()
        df[f'rolling_max_{window}'] = df[target_col].rolling(window, min_periods=1).max()

    df['trend'] = np.arange(len(df))
    df['ema_7'] = df[target_col].ewm(span=7, adjust=False).mean()
    df['ema_30'] = df[target_col].ewm(span=30, adjust=False).mean()

    return df

def train_prophet_model(df, target_col, model_name):
    """
    Train Prophet model and save it using pickle
    """
    prophet_df = df[['date', target_col]].rename(columns={'date':'ds','total':'y'}).copy()
    prophet_df.rename(columns={target_col:'y'}, inplace=True)

    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,
                    changepoint_prior_scale=0.05, seasonality_prior_scale=10)
    model.fit(prophet_df)

    # Save model
    with open(os.path.join(MODEL_DIR, f"{model_name}_prophet.pkl"), "wb") as f:
        pickle.dump(model, f)

    return model

def train_xgboost_model(train_df, val_df, target_col, model_name):
    """
    Train XGBoost model and save it using joblib
    """
    train_features = create_time_features(train_df, target_col)
    val_features = create_time_features(val_df, target_col)

    feature_cols = [c for c in train_features.columns if c not in ['date', target_col]]

    X_train = train_features[feature_cols].fillna(0)
    y_train = train_features[target_col]
    X_val = val_features[feature_cols].fillna(0)
    y_val = val_features[target_col]

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val,y_val)], verbose=False)

    joblib.dump(model, os.path.join(MODEL_DIR, f"{model_name}_xgb.pkl"))
    return model, feature_cols

def evaluate_model(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100 if mask.sum()>0 else 0
    return mae, rmse, mape

def update_ensemble_weights(collection, age_group, prophet_error, xgb_error):
    """
    Adjust ensemble weights inversely proportional to validation error
    """
    # Load existing weights
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE,'r') as f:
            weights = json.load(f)
    else:
        weights = {}

    if collection not in weights:
        weights[collection] = {}
    # Lower error → higher weight
    total_inv_error = (1/prophet_error) + (1/xgb_error)
    prophet_w = (1/prophet_error)/total_inv_error
    xgb_w = (1/xgb_error)/total_inv_error
    weights[collection][age_group] = {"prophet": prophet_w, "xgb": xgb_w}

    with open(WEIGHTS_FILE,'w') as f:
        json.dump(weights, f, indent=4)

# ============================ FORECAST FUNCTIONS ============================

def forecast_backend(state, district, collection, age_group, forecast_days=30):
    """
    Returns JSON-ready forecast for backend use
    """
    df, target_cols = load_collection_data(COLLECTIONS[collection]["collection_name"],
                                           state, district, collection)
    if age_group not in target_cols:
        raise ValueError(f"{age_group} not in target columns for {collection}")

    # Train-validation split
    split_date = df['date'].max() - timedelta(days=30)
    train_df = df[df['date'] <= split_date].copy()
    val_df = df[df['date'] > split_date].copy()

    model_name = f"{collection}_{age_group}_{state}_{district if district else 'ALL'}"

    # Load or train models
    prophet_path = os.path.join(MODEL_DIR, f"{model_name}_prophet.pkl")
    xgb_path = os.path.join(MODEL_DIR, f"{model_name}_xgb.pkl")

    if os.path.exists(prophet_path):
        with open(prophet_path,'rb') as f:
            prophet_model = pickle.load(f)
    else:
        prophet_model = train_prophet_model(train_df, age_group, model_name)

    if os.path.exists(xgb_path):
        xgb_model = joblib.load(xgb_path)
        feature_cols = [c for c in create_time_features(train_df, age_group).columns if c not in ['date', age_group]]
    else:
        xgb_model, feature_cols = train_xgboost_model(train_df, val_df, age_group, model_name)

    # Ensemble weights
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE,'r') as f:
            weights = json.load(f)
        w = weights.get(collection, {}).get(age_group, {"prophet":0.5, "xgb":0.5})
    else:
        w = {"prophet":0.5, "xgb":0.5}

    # Prophet forecast
    future_prophet = prophet_model.make_future_dataframe(periods=forecast_days)
    prophet_forecast = prophet_model.predict(future_prophet)['yhat'].values[-forecast_days:]
    prophet_forecast = np.maximum(prophet_forecast,0)

    # XGBoost forecast (recursive)
    xgb_future = []
    current_df = df.copy()
    future_dates = pd.date_range(df['date'].max()+timedelta(days=1), periods=forecast_days)
    for future_date in future_dates:
        temp_df = pd.concat([current_df, pd.DataFrame({'date':[future_date], age_group:[0]})], ignore_index=True)
        temp_features = create_time_features(temp_df, age_group)
        X_future = temp_features[feature_cols].fillna(0).iloc[-1:]
        pred = max(xgb_model.predict(X_future)[0],0)
        xgb_future.append(pred)
        current_df.loc[len(current_df)] = [future_date]+[pred if c==age_group else 0 for c in temp_df.columns[1:]]

    xgb_future = np.array(xgb_future)

    # Ensemble
    ensemble_forecast = w['prophet']*prophet_forecast + w['xgb']*xgb_future
    ensemble_forecast = np.maximum(ensemble_forecast,0)

    # Return JSON-ready dict
    return {"dates": future_dates.strftime("%Y-%m-%d").tolist(),
            "ensemble_forecast": ensemble_forecast.round(0).astype(int).tolist(),
            "prophet_forecast": prophet_forecast.round(0).astype(int).tolist(),
            "xgb_forecast": xgb_future.round(0).astype(int).tolist()}

def forecast_frontend(state, district, collection, age_group, forecast_days=30):
    """
    Returns DataFrame for plotting: historical, validation, forecast
    """
    df, target_cols = load_collection_data(COLLECTIONS[collection]["collection_name"],
                                           state, district, collection)
    if age_group not in target_cols:
        raise ValueError(f"{age_group} not in target columns for {collection}")

    backend_forecast = forecast_backend(state, district, collection, age_group, forecast_days)
    forecast_df = pd.DataFrame({
        "date": pd.to_datetime(backend_forecast["dates"]),
        "ensemble": backend_forecast["ensemble_forecast"],
        "prophet": backend_forecast["prophet_forecast"],
        "xgb": backend_forecast["xgb_forecast"]
    })
    return pd.concat([df[['date', age_group]].rename(columns={age_group:"historical"}), forecast_df], ignore_index=False)

# ============================ TRAIN ALL MODELS ============================

def train_all_models(states_list, districts_dict=None):
    """
    Train Prophet + XGBoost models for all collections and age groups
    districts_dict: {"Uttar Pradesh": ["Gorakhpur", "Lucknow"], ...}
    """
    for collection, info in COLLECTIONS.items():
        collection_name = info['collection_name']
        targets = info['targets']
        for state in states_list:
            districts = districts_dict.get(state, [None]) if districts_dict else [None]
            for district in districts:
                df, target_cols = load_collection_data(collection_name, state, district, collection)
                if df is None:
                    continue
                for age_group in target_cols:
                    # Special handling for enrollment: combine 5_17 + 18_plus
                    if collection=="enrollment" and age_group=="age_5_18_plus":
                        df[age_group] = df['mertics'].apply(lambda x: x.get('enroll_5_17',0)+x.get('enroll_18_plus',0) if isinstance(x, dict) else 0)
                    split_date = df['date'].max()-timedelta(days=30)
                    train_df = df[df['date']<=split_date]
                    val_df = df[df['date']>split_date]

                    model_name = f"{collection}_{age_group}_{state}_{district if district else 'ALL'}"

                    # Train Prophet
                    prophet_model = train_prophet_model(train_df, age_group, model_name)
                    # Prophet validation
                    future_val = pd.DataFrame({'ds':val_df['date']})
                    y_pred_prophet = np.maximum(prophet_model.predict(future_val)['yhat'].values,0)
                    y_true = val_df[age_group].values
                    prophet_mae, prophet_rmse, prophet_mape = evaluate_model(y_true, y_pred_prophet)

                    # Train XGBoost
                    xgb_model, feature_cols = train_xgboost_model(train_df, val_df, age_group, model_name)
                    val_features = create_time_features(val_df, age_group)
                    X_val = val_features[feature_cols].fillna(0)
                    y_pred_xgb = np.maximum(xgb_model.predict(X_val),0)
                    xgb_mae, xgb_rmse, xgb_mape = evaluate_model(y_true, y_pred_xgb)

                    # Update ensemble weights
                    update_ensemble_weights(collection, age_group, prophet_mae, xgb_mae)

                    print(f"✅ Trained {collection} | {age_group} | {state} | {district if district else 'ALL'}")
                    print(f"   Prophet MAE: {prophet_mae:.2f}, XGB MAE: {xgb_mae:.2f}")

# ============================ END OF PIPELINE ============================
print("Pipeline ready. You can now call `train_all_models()`, `forecast_backend()` or `forecast_frontend()`")
