import pandas as pd

def fetch_data(query, conn):
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            columns = [desc[0] for desc in cur.description]
            data = cur.fetchall()
            return pd.DataFrame(data, columns=columns)
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return pd.DataFrame()