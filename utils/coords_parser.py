import numpy as np
import pandas as pd
import torch


def intify_atlante(data):
    if 'W' in data or 'S' in data:
        return -float(data[:-1])
    elif 'E' in data or 'N' in data:
        return float(data[:-1])


def intify_pacifique(data):
    if 'W' in data or 'S' in data:
        return 360 - float(data[:-1])
    elif 'E' in data or 'N' in data:
        return float(data[:-1])


def intify(data, ocean):
    if 'pacific' in ocean:
        return intify_pacifique(data)
    else:
        return intify_atlante(data)


def coords_parser(filename):
    """
    Parses NOAA csv file to extract the coordinates of the hurricane/ storm during its progression.

    :param filename: Path to the csv file
    :return A dictionnary with athe Hurricane ID as key, and a list of its coordinates.
    Each coordinate is a list of two float values
    """
    coords_dict = {}
    csvfile = pd.read_csv(filename)

    for record_idx in range(csvfile.shape[0]):
        _key = csvfile.loc[record_idx, 'ID']
        value_lon = (
            intify(csvfile.loc[record_idx, 'Longitude'], filename) + 65
        ) / 20
        value_lat = (
            intify(csvfile.loc[record_idx, 'Latitude'], filename) - 27
        ) / 10

        try:
            coords_dict[_key].append([value_lat, value_lon])
        except KeyError:
            coords_dict[_key] = [[value_lat, value_lon]]

    for key in coords_dict:
        coords_dict[key] = torch.tensor(
            np.array(coords_dict[key]), dtype=torch.float32
        )
    return coords_dict
