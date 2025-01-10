import requests
from typing import List, Dict, Tuple
import pandas as pd
import time
from tqdm import tqdm

def test_medical_property(property_id: str, example_entities: List[str], entity_type: str = 'Q12136') -> None:
    """
    Test a medical property with multiple example entities to see response patterns
    """
    for entity in example_entities:
        query = f"""
        SELECT DISTINCT ?entity ?entityLabel ?value ?valueLabel
        WHERE {{
          ?entity wdt:P31/wdt:P279* wd:{entity_type};
                  rdfs:label "{entity}"@en;
                  wdt:{property_id} ?value.
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "en".
            ?entity rdfs:label ?entityLabel.
            ?value rdfs:label ?valueLabel.
          }}
        }}
        LIMIT 3
        """
        
        results = query_wikidata(query)
        
        if results:
            print(f"\nResults for {entity}:")
            for result in results:
                print(f"- {result['valueLabel']['value']}")
        else:
            print(f"\nNo results found for {entity}")
        
        time.sleep(1)  # Rate limiting

if __name__ == "__main__":
    # Test different medical properties
    print("\nTesting route of administration (P636) for medications:")
    test_medical_property("P636", ["Aspirin", "Insulin"], "Q12140")
    
    print("\nTesting anatomical location (P927) for diseases:")
    test_medical_property("P927", ["Pneumonia", "Hepatitis"])
    
    print("\nTesting diagnostic method (P1995) for diseases:")
    test_medical_property("P1995", ["Tuberculosis", "Diabetes"])
    
    print("\nTesting drug mechanism of action (P3776) for medications:")
    test_medical_property("P3776", ["Warfarin", "Metformin"], "Q12140")