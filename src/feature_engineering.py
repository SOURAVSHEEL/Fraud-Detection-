import pandas as pd
import numpy as np

def create_time_features(df):
    """Creating time-based features from transaction data"""
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = df['hour'].between(23, 5).astype(int)
    
    # Calculate transaction frequency
    df['transactions_last_hour'] = df.groupby('customer_id')['timestamp'].transform(
            lambda x: x.rolling('1H').count()
    )
    return df

def create_location_features(df):
    """Generating location-based risk scores"""
    # Calculate state-wise fraud rates
    state_risk = df.groupby('merchant_state')['is_fraud'].mean()
    df['state_risk_score'] = df['merchant_state'].map(state_risk)
        
    # Calculate distance between customer and merchant
    df['transaction_distance'] = np.sqrt(
        (df['customer_lat'] - df['merchant_lat'])**2 +
        (df['customer_lon'] - df['merchant_lon'])**2
    )
    return df
