# Wikidata Dataset
This folder contains the files needed to create the medical Q&A dataset for entity recognition, using wikidata.
The files are divided as so:
- `main.py`: Script for generating the dataset from queries to Wikidata.
- `entity_recognition_dataset.csv`: extracted dataset with queries for all the entity names and properties extracted.
- `formatted_entity_recognition_dataset.csv`: filtered from `entity_recognition_dataset.csv` to have 2 questions per entity, and correspond with the format used on the other dataset.
- `testing.py`: simple testing framework for finding properties and entities.
- `dataset_exploration.ipynb`: notebook for exploring the dataset, and used to format and filter the `formatted_entity_recognition_dataset.csv` file.
