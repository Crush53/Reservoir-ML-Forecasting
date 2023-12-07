import pandas as pd 
def getData(stn, vals):
    try:
        # Extract time series data
        ts = vals['value']['timeSeries'][-1]['values'][-1]['value']
        
        # Convert to DataFrame
        df = pd.DataFrame(ts)
        
        # Ensure the 'value' column is float type
        df['value'] = df['value'].astype(float)
        
        #print some basic stats about the data to check if it was recieved
        print(df.describe())
        
        return df
    except Exception as e:
        print(f"Error parsing data for station {stn}: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame on error   