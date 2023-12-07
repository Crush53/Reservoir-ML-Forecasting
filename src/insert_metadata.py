import os
import psycopg2
import pandas as pd  # Ensure pandas is used elsewhere in the file
from dotenv import load_dotenv
load_dotenv()

def insert_metadata(data):
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASS"),
            sslmode="require"
        )
        cur = conn.cursor()
        sql = """INSERT INTO res_meta (stn_id, stn_name, lat, lon) VALUES (%s, %s, %s, %s) ON CONFLICT (stn_id) DO NOTHING;"""
        cur.execute(sql, (data['stn_id'], data['stn_name'], data['lat'], data['lon']))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error inserting metadata: {str(e)}")