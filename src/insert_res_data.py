import os
import psycopg2
import pandas as pd  # Ensure pandas is used elsewhere in the file
from dotenv import load_dotenv
load_dotenv()

def insert_res_data(stn_id, df):
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASS"),
            sslmode="require"
        )
        print(df.head())
        cur = conn.cursor()
        sql = """INSERT INTO res_data (stn_id, datetime, storage_value) VALUES (%s, %s, %s);"""
        for index, row in df.iterrows():
            cur.execute(sql, (stn_id, row['dateTime'], row['value']))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error inserting reservoir data: {str(e)}")