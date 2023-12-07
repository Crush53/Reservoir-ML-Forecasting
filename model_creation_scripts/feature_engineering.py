import pandas as pd

def create_lag_features(df, lag_days, column_name):
    for i in range(1, lag_days + 1):
        df[f'{column_name}_lag_{i}'] = df[column_name].shift(i)
    return df