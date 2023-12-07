from dotenv import load_dotenv
load_dotenv()
import os
import psycopg2
import pandas as pd
def create_tables():
    commands = (
         """
        CREATE TABLE IF NOT EXISTS res_meta (
            stn_id VARCHAR(10) PRIMARY KEY,
            stn_name VARCHAR(255),
            lat FLOAT,
            lon FLOAT
        );
        """,
        """ 
        CREATE TABLE IF NOT EXISTS res_data (
            id SERIAL PRIMARY KEY,
            stn_id VARCHAR(10) REFERENCES res_meta(stn_id),
            datetime TIMESTAMP,
            storage_value FLOAT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS weather_data (
            id SERIAL PRIMARY KEY,
            stn_id VARCHAR(10) REFERENCES res_meta(stn_id),
            datetime TIMESTAMP,
            average_temperature FLOAT,
            precipitation FLOAT
        );
        """
    )
    conn = None
    try:
        # connect to the server
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASS"),
            sslmode="require"
        )
        cur = conn.cursor()
        # execute each command
        for command in commands:
            cur.execute(command)
        # close communication with the database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()