def getMeta(stn, vals):
    vars=['siteName','geoLocation']
    if len(vals['value']['timeSeries']):
        for var in vars:
            if var == 'geoLocation':
                lat=vals['value']['timeSeries'][-1]['sourceInfo'][var]['geogLocation']['latitude']
                lon=vals['value']['timeSeries'][-1]['sourceInfo'][var]['geogLocation']['longitude']
            else:
                stn_name=vals['value']['timeSeries'][-1]['sourceInfo'][var]
        return {'stn_id': stn, 'stn_name': stn_name, 'lat': lat, 'lon': lon}
    else:
        return {'stn_id': stn, 'stn_name': None, 'lat': None, 'lon': None}