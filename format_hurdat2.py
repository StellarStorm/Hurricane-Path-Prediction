import os

import pandas as pd
import requests

ATL_HURDAT2 = (
    'https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt'
)

columns = [
    'ID',
    'Name',
    'Date',
    'Time',
    'Event',
    'Status',
    'Latitude',
    'Longitude',
    'Maximum Wind',
    'Minimum Pressure',
    'Low Wind NE',
    'Low Wind SE',
    'Low Wind SW',
    'Low Wind NW',
    'Moderate Wind NE',
    'Moderate Wind SE',
    'Moderate Wind SW',
    'Moderate Wind NW',
    'High Wind NE',
    'High Wind SE',
    'High Wind SW',
    'High Wind NW',
    'Maximum Wind Radius',
]

records = []
response = requests.get(ATL_HURDAT2)
lines = response.text.split('\n')

for line in lines:
    components = [x.strip() for x in line.split(',')]
    if components[0].startswith('AL'):
        stormid = components[0]
        name = components[1]
    else:
        if len(components[0]) > 0:
            records.append([stormid, name] + components)

hurdat2_csv = pd.DataFrame(records, columns=columns)

hurdat2_csv.to_csv(os.path.join('Data', 'atlantic_latest.csv'), index=False)
