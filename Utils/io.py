import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


base_path = Path(__file__).parent.parent

def read_parameters_from_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def read_parameters_from_txt(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split(':')
                parameters[key.strip()] = value.strip()
                print(f"{key.strip()}: {value.strip()}")
    return parameters


def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' deleted successfully.")
    except FileNotFoundError:
        print(f"Folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred while deleting folder '{folder_path}': {e}")


def prepare_hvac_datasheet(HVAC_file_name):
    folder_path = base_path.joinpath('sim_hvac_output')
    if not os.path.exists(folder_path.joinpath(f'{HVAC_file_name}.csv')):
        columns = ['date', 'time', 'outdoor temperature',
                   'indoor temperature 1', 'indoor temperature 2', 'indoor temperature 3',
                   'indoor temperature 4', 'indoor temperature 5', 'indoor temperature 6',
                   'EHVAC']

        df = pd.DataFrame(columns=columns)

        # Save the updated DataFrame back to the CSV file
        df.to_csv(folder_path.joinpath(f'{HVAC_file_name}.csv'), sep=',', index=False)
        print("CSV file generated successfully.")


def output_to_csv(file_path, data):
    # Read the CSV file
    df = pd.read_csv(file_path, delimiter=',')

    # Append the new variables and values to the bottom line
    new_row = pd.DataFrame({
        'date': data['date'],
        'time': data['time'],
        'outdoor temperature': data['outdoor temperature'],
        'indoor temperature 1': data['indoor temperature 1'],
        'indoor temperature 2': data['indoor temperature 2'],
        'indoor temperature 3': data['indoor temperature 3'],
        'indoor temperature 4': data['indoor temperature 4'],
        'indoor temperature 5': data['indoor temperature 5'],
        'indoor temperature 6': data['indoor temperature 6'],
        'EHVAC': data['EHVAC'],

    })
    # df = df.append(new_row, ignore_index=True)
    df = pd.concat([df, new_row])

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, sep=',', index=False)
