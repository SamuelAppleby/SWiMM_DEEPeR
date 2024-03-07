import os
import json


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {file_path}: {e}")
        return None


def process_all_json_files(directory_path):
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]

    for json_file in json_files:
        file_path = os.path.join(directory_path, json_file)
        json_data = read_json_file(file_path)

        if json_data['msgType'] == 'rolloutEnd':
            x = 5
        if json_data:
            print(f"JSON data from {json_file}:")
            print(json_data)
            print()


directory_path = 'C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR/logs/sac/sac_3/network/training/episode_1/packets_sent'
process_all_json_files(directory_path)
