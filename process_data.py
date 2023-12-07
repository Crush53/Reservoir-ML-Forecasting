import requests as rq
import json
import pandas as pd
import src.create_tables as ct
import src.ingest_resevoir_data as ird
import src.ingest_resevoir_meta as irm
import psycopg2
import os
# Function to fetch weather data from API 
def fetch_weather_data(parms):
    url = "https://data.rcc-acis.org/GridData"
    response = rq.post(url, data=json.dumps(parms), headers={'content-type': 'application/json'}, timeout=60)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None
    
#function to insert weather data into database    
def insert_weather_data(df):
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASS"),
            sslmode="require"
        )
        cur = conn.cursor()
        sql = """INSERT INTO weather_data (stn_id, datetime, average_temperature, precipitation) VALUES (%s, %s, %s, %s);"""
        for index, row in df.iterrows():
            cur.execute(sql, (row['stn_id'], row['datetime'], row['Average Temperature'], row['Precipitation']))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error inserting weather data: {str(e)}")

#function to process and insert weather data into database
# Uses insert_weather_data function ,get_station_id_from_params function and fetch_weather_data function
def process_and_insert_weather_data(parms):
    # Fetch data from API
    weather_data = fetch_weather_data(parms)
    station_id = get_station_id_from_params(parms)
    
    # Specify column names for the DataFrame
    column_names = ['datetime', 'Average Temperature', 'Precipitation']

    # Convert to DataFrame
    weather_df = pd.DataFrame(weather_data['data'], columns=column_names) if weather_data else pd.DataFrame(columns=column_names)

    # Add 'stn_id' and convert 'datetime'
    weather_df['stn_id'] = station_id
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
    weather_df = weather_df[['stn_id', 'datetime', 'Average Temperature', 'Precipitation']]

    # Insert into database
    insert_weather_data(weather_df)
    
#function to get station id from parameters from extracting the longitude and latitude from the loc parameter
def get_station_id_from_params(params):
    try:
        conn = psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        dbname=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASS"),
        sslmode="require"
        )
        # Extract longitude and latitude from the 'loc' parameter
        lon, lat = map(float, params["loc"].split(','))
        
        with conn.cursor() as cur:
            sql = """SELECT stn_id FROM res_meta WHERE lon = %s AND lat = %s;"""
            cur.execute(sql, (lon, lat))  # Note the order: (longitude, latitude)
            result = cur.fetchone()
            return result[0] if result else None
    except Exception as e:
        print(f"Error retrieving station ID: {str(e)}")
        return None


def process_data():
    #parameters for the weather API 
    parms_alisa_res = {
        "loc": "-120.1301473,34.54887705",
        "sdate": "20100101",
        "edate": "20231031",
        "grid": "21",
        "elems": [
            {"name": "avgt", "interval": "dly", "duration": "dly"},
            {"name": "pcpn", "interval": "dly", "duration": "dly"}
        ]
    }
    parms_santa_ynez = {
        "loc": "-119.6865245,34.52610567", 
        "sdate": "20100101",  
        "edate": "20231031",  
        "grid": "21",  
        "elems": [
            {"name": "avgt", "interval": "dly", "duration": "dly"},  # Average temperature
            {"name": "pcpn", "interval": "dly", "duration": "dly"}  # Precipitation
        ]
    }

    # Fetch data from API for weather for Alisa Reservoir and Santa Ynez Reservoir
    #creating the tables (res_meta,res_data,weather_data) and ingesting the data into my database
    ct.create_tables()
    irm.ingest_resevoir_meta()
    ird.ingest_resevoir_data()
    process_and_insert_weather_data(parms_alisa_res)
    process_and_insert_weather_data(parms_santa_ynez)

if __name__ == 'process_data':
    process_data()  
    


