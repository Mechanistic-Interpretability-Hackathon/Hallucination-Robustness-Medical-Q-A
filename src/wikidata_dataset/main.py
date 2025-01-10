import requests
from typing import List, Dict, Tuple
import pandas as pd
import time
from tqdm import tqdm

def create_sparql_query(entity_type: str, property_id: str) -> str:
    """
    Create a SPARQL query for Wikidata to get entities and their single-value properties.
    Added GROUP_CONCAT and SAMPLE to handle multiple values but select one.
    """
    return f"""
    SELECT DISTINCT ?entity ?entityLabel (SAMPLE(?value) as ?singleValue) (SAMPLE(?valueLabel) as ?singleValueLabel)
    WHERE {{
      ?entity wdt:P31/wdt:P279* wd:{entity_type};
              wdt:{property_id} ?value.
      SERVICE wikibase:label {{ 
        bd:serviceParam wikibase:language "en".
        ?entity rdfs:label ?entityLabel.
        ?value rdfs:label ?valueLabel.
      }}
    }}
    GROUP BY ?entity ?entityLabel
    LIMIT 1000
    """

def query_wikidata(query: str) -> List[Dict]:
    """
    Query Wikidata using their SPARQL endpoint.
    Includes rate limiting and error handling.
    """
    endpoint_url = "https://query.wikidata.org/sparql"
    
    try:
        response = requests.get(endpoint_url, 
                              params={'query': query, 'format': 'json'},
                              headers={'User-Agent': 'Medical Dataset Builder 1.0'})
        response.raise_for_status()
        return response.json()['results']['bindings']
    except Exception as e:
        print(f"Error querying Wikidata: {e}")
        return []

def create_templates(results: List[Dict], entity_type_name: str, relation_name: str) -> List[Dict]:
    """
    Convert Wikidata results into natural language templates.
    Only includes entries with single-value answers.
    """
    templates = []
    
    for result in results:
        try:
            entity_name = result['entityLabel']['value']
            attribute = result['singleValueLabel']['value']
            
            # Skip if attribute is too long (likely multiple values combined)
            if len(attribute.split()) > 3:
                continue
                
            template = {
                'entity_type': entity_type_name,
                'entity_name': entity_name,
                'relation': relation_name,
                'template': f"The {entity_type_name} {entity_name} {relation_name} ",
                'correct_answer': attribute
            }
            templates.append(template)
        except KeyError as e:
            continue
            
    return templates

def test_single_entity(entity_name: str, entity_type: str = 'Q12136', property_id: str = 'P61') -> None:
    """
    Test the retrieval for a single entity with single-value properties
    """
    query = f"""
    SELECT DISTINCT ?entity ?entityLabel ?value ?valueLabel
    WHERE {{
      ?entity wdt:P31/wdt:P279* wd:{entity_type};
              rdfs:label "{entity_name}"@en;
              wdt:{property_id} ?value.
      SERVICE wikibase:label {{ 
        bd:serviceParam wikibase:language "en".
        ?entity rdfs:label ?entityLabel.
        ?value rdfs:label ?valueLabel.
      }}
    }}
    LIMIT 5
    """
    
    results = query_wikidata(query)
    
    if not results:
        print(f"No results found for {entity_name}")
        return
        
    print(f"\nResults for {entity_name}:")
    for result in results:
        print(f"- {result['valueLabel']['value']}")

def build_medical_dataset():
    """
    Build a dataset of medical entities and their single-value relationships.
    """
    # Define entity types and their properties to query
    # Focus on properties that typically have single values
    queries = [
        # ('Q12136', 'P828', 'disease', 'is caused by'),  # Doesn't work well
        ('Q12136', 'P61', 'disease', 'was first identified by'),
        ('Q12136', 'P291', 'disease', 'originated in'),
        ('Q12136', 'P2293', 'disease', 'is caused by a mutation in the gene named'),
        ('Q12136', 'P927', 'disease', ' is located (anatomically) in the'),
        ('Q12140', 'P2275', 'medication', 'has an active ingredient with the name of'),
        ('Q12140', 'P2175', 'medication', 'is used to treat'),
        ('Q12140', 'P274', 'medication', ", it's chemical formula is"),
        ('Q12136', 'P780', 'disease', ', its main symptom is')
    ]
    all_templates = []
    
    for entity_type, property_id, type_name, relation in tqdm(queries):
        query = create_sparql_query(entity_type, property_id)
        results = query_wikidata(query)
        templates = create_templates(results, type_name, relation)
        all_templates.extend(templates)
        time.sleep(1)  # Rate limiting
    
    # Convert to DataFrame and remove duplicates
    df = pd.DataFrame(all_templates)
    df = df.drop_duplicates(subset=['template'])
    
    # Filter out templates with long answers
    df = df[df['correct_answer'].str.split().str.len() <= 3]
    
    return df

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
    # Test a single entity
    # print("Testing single entity retrieval...")
    # test_single_entity("COVID-19", "Q12136", "P780")
    
#     print("\nTesting transmition process:")
#     test_medical_property("P1060", ["COVID-19", "Diabetes"], "Q12136")
#     print("\nTesting causes for disease:")
#     test_medical_property("P828", ["COVID-19", "common cold"], "Q12136")

    print("\nBuilding complete medical dataset from Wikidata...")
    dataset = build_medical_dataset()

    # Save to CSV
    dataset.to_csv('entity_recognition_dataset.csv', index=False)
    print(f"\nDataset created with {len(dataset)} templates")
    print("\nSample templates:")
    print(dataset[['template', 'correct_answer']].head())
