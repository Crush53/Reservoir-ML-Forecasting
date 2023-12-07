import requests
import json
# Import necessary functions from other files in src
from .getData import getData
from .insert_res_data import insert_res_data

def ingest_resevoir_data():
    # Fetch, parse, and insert data for each URL for resdata table    
    # URLs to fetch data
    urls = [
    "https://nwis.waterservices.usgs.gov/nwis/dv/?format=json&sites=11122000&startDT=2010-01-01&endDT=2023-10-31&parameterCd=00054&siteType=LK&siteStatus=all",
    "https://nwis.waterservices.usgs.gov/nwis/dv/?format=json&sites=11128300&startDT=2010-01-01&endDT=2023-10-31&parameterCd=00054&siteType=LK&siteStatus=all"
    ]
    
    for url in urls:
        try:
            response = requests.get(url)
            vals = json.loads(response.text)
            stn_id = url.split("&sites=")[1].split("&")[0]
            
            # Parse and insert data
            data_df = getData(stn_id, vals)
            insert_res_data(stn_id, data_df)
        except Exception as e:
            print(f"Error processing URL {url}: {str(e)}")

