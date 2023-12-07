import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def connect_database():
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASS"),
            sslmode="require"
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None