import pandas as pd
from .feature_engineering import create_lag_features


def prepare_dataset(df_reservoir, df_weather, lag_days):
    # Remove duplicates from df_weather
    df_weather = df_weather.drop_duplicates(subset=['stn_id', 'datetime'])

    # Merge the data on 'stn_id' and 'datetime'
    df_combined = pd.merge(df_reservoir, df_weather, on=['stn_id', 'datetime'], how='inner')

    # Convert 'datetime' to datetime object and sort the dataset by date
    df_combined['datetime'] = pd.to_datetime(df_combined['datetime'])
    df_combined.sort_values(by='datetime', inplace=True)

    predictor_columns = [f'storage_value_lag_{i}' for i in range(1, lag_days + 1)] + ['average_temperature', 'precipitation']
    
    # Group the DataFrame by 'stn_id' and apply the lag feature creation
    grouped = df_combined.groupby('stn_id')
    
    df_list = [create_lag_features(group, lag_days, 'storage_value') for name, group in grouped]

    # Concatenate the groups back together
    df_combined = pd.concat(df_list)
    
    # Drop rows with NaN values created due to lagging
    df_combined.dropna(subset=[f'storage_value_lag_{i}' for i in range(1, lag_days + 1)], inplace=True)
    
    # Reset index to get datetime back as a column
    df_combined.reset_index(inplace=True, drop=True)
    
    return df_combined, predictor_columns

