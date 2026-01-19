import pandas as pd
from prophet import Prophet
from app.utils.database import biometric_collection,demographic_collection,enrollment_collection

# 1. Load data from MongoDB into a DataFrame
# (Assuming you've fetched the 'metrics' and 'date' fields)
df = pd.DataFrame(list(biometric_collection.find()))
# 2. Flatten the 'metrics' column so bio_age_5_17 etc. become real columns
df_metrics = pd.json_normalize(df['metrics'])
df = pd.concat([df.drop('metrics', axis=1), df_metrics], axis=1)

# 3. Ensure date is a datetime object and sort
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# 4. Fill missing days (Crucial for Prediction)
# If a pincode has no data on Tuesday, models need to see a "0" there.
df = df.set_index('date').groupby('pincode').resample('D').asfreq().fillna(0)


def predict_next_30_days(pincode_data):
    # Prophet requires columns 'ds' (date) and 'y' (value)
    model_df = pincode_data[['date', 'bio_age_5_17']].rename(columns={'date': 'ds', 'bio_age_5_17': 'y'})
    
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(model_df)
    
    # Create a dataframe for the next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]