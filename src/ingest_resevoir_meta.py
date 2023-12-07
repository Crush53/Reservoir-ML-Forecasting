import requests
import json
# Import necessary functions from other files in src
from .getMeta import getMeta
from .insert_metadata import insert_metadata


def ingest_resevoir_meta():
# URLs to fetch data
    urls = [
    "https://nwis.waterservices.usgs.gov/nwis/dv/?format=json&sites=11122000&startDT=2010-01-01&endDT=2023-10-31&parameterCd=00054&siteType=LK&siteStatus=all",
    "https://nwis.waterservices.usgs.gov/nwis/dv/?format=json&sites=11128300&startDT=2010-01-01&endDT=2023-10-31&parameterCd=00054&siteType=LK&siteStatus=all"
    ]
    # Fetch, parse, and insert metadata for each URL for metadata table
    for url in urls:
        try:
            response = requests.get(url)
            vals = json.loads(response.text)
            stn_id = url.split("&sites=")[1].split("&")[0]
            
            # Parse and insert metadata
            meta_data = getMeta(stn_id, vals)
            insert_metadata(meta_data)
        except Exception as e:
            print(f"Error processing URL {url}: {str(e)}")
            # Fetch, parse, and insert data for each URL
