import pandas as pd
import goodfire
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

GOODFIRE_API_KEY = os.getenv("GOODFIRE_API_KEY")


def get_feature_activations_v2(
    client: goodfire.AsyncClient,
    variant: goodfire.Variant,
    entities: pd.DataFrame,
    known_unknown_filter: str,
    key_name: str = "query",
    k: int = 100,
):
    """
    Processes a list of dictionaries, filtering by the 'known_unknown' key
    and operating on the 'key_name' key for feature activation retrieval, default to get features for the query
    """
    feature_activations = []
    feature_library = set()

    # Filter entities by the 'known_unknown' key
    filtered_entities = [
        entity
        for entity in entities
        if entity.get("known_unknown") == known_unknown_filter
    ]

    for record in tqdm(filtered_entities, desc="Processing entities"):
        try:
            item = record.get(key_name)
            if not item:
                print(f"Skipping item : {record}")
                continue

            inspector = client.features.inspect(
                [{"role": "user", "content": item}],
                model=variant,
            )

            # Prepare to collect activated features
            features = []

            for activation in inspector.top(k=k):
                # Keep our own library of all unique features and their IDs
                feature_library.add((activation.feature.uuid, activation.feature.label))

                # Record the feature ID and its activation value for this query
                features.append(
                    {
                        "uuid": activation.feature.uuid,
                        "activation": activation.activation,
                    }
                )

            feature_activations.append(
                {
                    "query": record.get("query"),
                    "entity": record.get("entity"),
                    "known_unknown": record.get("known_unknown"),
                    "red_herring": record.get("red_herring"),
                    "entity_features": features,
                }
            )

        except Exception as e:
            print(f"Failed to process entity: {str(e)}")
            continue

    return feature_activations, feature_library


if __name__ == "__main__":
    # load data
    queries_for_feature_extraction = pd.read_parquet(
        "./queries_for_feature_extraction.parquet"
    )

    # convert from pandas to list of dicts
    queries_for_feature_extraction.to_dict("records")

    # insepct sample
    queries_for_feature_extraction[0:4]
    client_gf = goodfire.Client(GOODFIRE_API_KEY)
    variant = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    feature_activations_known_1Q, feature_library_known_1Q = get_feature_activations_v2(
        client_gf,
        variant,
        queries_for_feature_extraction,
        known_unknown_filter="known",
        key_name="query",
        k=100,
    )
