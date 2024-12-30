import goodfire
import json
import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

def save_features(features: goodfire.FeatureGroup, json_path):
    features_data = features.json()
    with open(json_path, 'w') as f:
        json.dump(features_data, f, indent=4)

def load_features(json_path) -> goodfire.FeatureGroup:
    with open(json_path, 'r') as f:
        features_data = json.load(f)
        loaded_features = goodfire.FeatureGroup.from_json(features_data)
    return loaded_features

def process_jsonl_file(file_path):
    with open(file_path) as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]
    df = pd.DataFrame(data)
    return df