import numpy as np
import pandas as pd
import torch


def normalize_coords(coord, ocean='atlantic'):
    hemisphere = coord[-1]
    stripped_cord = float(coord[:-1])

    if hemisphere not in ('N', 'S', 'E', 'W'):
        raise ValueError(
            f'Coordinates must end in one of N/S/E/W, not {hemisphere}'
        )

    if hemisphere == 'N':
        return (stripped_cord - 27) / 10
    if hemisphere == 'E':
        return (stripped_cord + 65) / 20
    if hemisphere == 'S':
        if ocean == 'atlantic':
            return (-stripped_cord - 27) / 10
        return ((360 - stripped_cord) - 27) / 10
    if hemisphere == 'W':
        if ocean == 'atlantic':
            return (-stripped_cord + 65) / 20
        return (360 - stripped_cord + 65) / 20


def coords_parser(filename, ocean: str = 'auto'):
    """
    Parses NOAA csv file to extract the coordinates of the hurricane/ storm during its progression.

    :param filename: Path to the csv file
    :return A dictionnary with athe Hurricane ID as key, and a list of its coordinates.
    Each coordinate is a list of two float values
    """
    coords_dict = {}
    csvfile = pd.read_csv(filename)
    if ocean == 'auto':
        if 'atlantic' in filename.lower():
            ocean = 'atlantic'
        else:
            ocean = 'pacific'

    for record_idx in range(csvfile.shape[0]):
        _key = csvfile.loc[record_idx, 'ID']
        value_lon = normalize_coords(
            csvfile.loc[record_idx, 'Longitude'], ocean=ocean
        )
        value_lat = normalize_coords(
            csvfile.loc[record_idx, 'Latitude'], ocean=ocean
        )

        # Patricia max wind speed was 185 kt - using 200 as max
        if pd.isnull(csvfile.loc[record_idx, 'Maximum Wind']):
            winds = 52.6 / 200  # 52.6 is mean value
        else:
            winds = csvfile.loc[record_idx, 'Maximum Wind'] / 200
        # Wilma lowest pressure was 882 mb - using 850 as min
        # 1024 is the upper limit in the atlantic data
        if pd.isnull(csvfile.loc[record_idx, 'Minimum Pressure']):
            pressure = (1024 - 992) / (1024 - 850)  # 992 is mean value
        else:
            pressure = (1024 - csvfile.loc[record_idx, 'Minimum Pressure']) / (
                1024 - 850
            )

        try:
            coords_dict[_key].append([value_lat, value_lon, winds, pressure])
        except KeyError:
            coords_dict[_key] = [[value_lat, value_lon, winds, pressure]]

    for key in coords_dict:
        coords_dict[key] = torch.tensor(
            np.array(coords_dict[key]), dtype=torch.float32
        )
    return coords_dict
