# Oli's Update Fri Jan 10

Last Sunday I promised to generate data for us to follow Ferrando et al's paper on hallucination features.
This proved far more complex than I had anticipated, but in summary:

The new source I recommend, and have processed, is the Human Disease Ontology (HumanDO) from the OBO Foundation.
Their website is:
- https://disease-ontology.org

You can download their HumanDo.json file...
- Directly from the OBO foundation github site:
    - https://github.com/DiseaseOntology/HumanDiseaseOntology/tree/main/src/ontology

- Via VS Code, having pulled latest github
    - "../classifier/data_classifier/HumanDO.json"

- Or, directly from our github repo:
    - https://github.com/Mechanistic-Interpretability-Hackathon/Hallucination-Robustness-Medical-Q-A/blob/main/src/classifier/data_classifier/HumanDO.json

## Processed Data

I have processed the above data source into two questions per disease entity, which I use to score the LLM on known and unknown entities. The processed data can be found at:

- Via VS Code, having pulled latest github
    - "../classifier/data_classifier/humando_parsed_1Q_completed_tg.parquet"

- Or, directly from our github repo:
    - https://github.com/Mechanistic-Interpretability-Hackathon/Hallucination-Robustness-Medical-Q-A/blob/main/src/classifier/data_classifier/humando_parsed_1Q_completed_tg.parquet

## Why Problems?

Medical and disease questions are harder than movie and basketball questions. Harder to define and harder to answer. Using other sources of data the LLM got only 5% of entities right on two questions. This was not a fair representation of what the LLM knew, so could not be used for analysing hallucinations.

You can follow the process in the jupyter notebook at:

- Via VS Code, having pulled latest github
    - "../classifier/FerrandoEtAl_Data.ipynb"

- Or, directly from our github repo:
    - https://github.com/Mechanistic-Interpretability-Hackathon/Hallucination-Robustness-Medical-Q-A/blob/main/src/classifier/FerrandoEtAl_Data.ipynb

### Process TLDR;

Here's a high-level summary of the notebook's workflow:

- Initially attempted to use PubMed IDs to test LLM knowledge, but found this approach unsuitable as no models could reliably identify PMIDs.
- Switched to using the NCBI Disease dataset for named entity recognition, but found the dataset too small for meaningful analysis.
- Finally settled on using the Human Disease Ontology (HumanDO) dataset, which provided a larger collection of disease entities and descriptions.
- Created a process to generate two test questions per disease using GPT-4, where each question had a single-word answer.
- Used Together.ai's TURBO(quantised) version of LLM to answer these questions, as Goodfire API was hopelessly slow and crashed. 
- Using this LLM, scored responses as known (both answers correct), unknown (both incorrect), or uncertain (one correct).
- Used Goodfire's sparse autoencoder to analyze feature activations for both known and unknown disease entities.
- Calculated separation scores between features activated by known vs. unknown entities.
- Identified key features associated with entity unfamiliarity, particularly finding evidence of RLHF-trained rejection features for unknown entities.

This process largely followed Ferrando et al.'s methodology, but with adaptations to work with medical entity recognition rather than general knowledge testing.

## Results

See the end of the jupyter notebook (link above.)

Top 5 Features with Greatest Separation, Unknown vs Known Diseases:

| Label | Feature ID | Known Frac | Unknown Frac | Sep Score |
|-------|------------|------------|--------------|------------|
| Explanations of rare genetic disorders and syndromes | 1292651f-8c52-4863-acf8-615b472c95bf | 0.199434 | 0.356287 | -0.156853 |
| Requests for 2000-word technical articles about industrial topics | b0226353-caf5-41e8-a645-888e6ff42100 | 0.062235 | 0.183234 | -0.120999 |
| References to having or experiencing medical/psychological disorders | 7ac9f4da-3e62-48b8-8699-062101e362cb | 0.181047 | 0.294012 | -0.112965 |
| Classical and technical word suffixes, especially in long words | adbdc27d-53f2-4398-b82b-6291c3d59e2c | 0.222065 | 0.323353 | -0.101288 |
| Nonsensical or potentially harmful input requiring clarification or rejection | 475a1ebc-1432-4f32-a8f8-f3398de51bec | 0.214993 | 0.315569 | -0.100576 |

The above features are more common for unknown than known features, the fifth feature is especially interesting.

### FEATURE: 'Nonsensical or potentially harmful input requiring clarification or rejection'

This feature is as predicted by Ferrando et al for identifying unknown entites, in our case, diseases. Its an RLHF trained rejection of nonsensical inputs, as discussed in their paper. It enjoys a relatively high separation score between unknown and known entities in the dataset, ranked no. 5.

BUT, why is the separation score relatively low, just 10%?

This is likely an issue with the quality of separation of the data and the model used for chat completion (scoring):
- For example, breast cancer appears in the 'unknown' entites, but I find this hard to believe.
More likely, the breast cancer questions were simply too difficult to get perfectly correct.
- Furthermore, many of the 'known' entities may actually have been 'uncertain' rather than fully 'known', hence 21% of known entities activate this feature.
- Finally, the scoring was done using the Turbo version of LLM on Together.ai, because it is fast and reliable. But that version is quantised and does not perform the same as the standard model, which is used by Goodfire for the SAE. We'd have used Goodfire for scoring, but their chat completion API repeatedly failed after just 100 calls.

## Next Steps / What Could Be Improved
 
1. Improve the Data
- Review prompt used to create two questions from HumanDO data, propose better prompts to scoring to sift known form unknown diseases.

2. Use the Proper LLM for Scoring.
- Would be great if we can use the Goodfire chat completion API to answer the questions, so we ar eusing precisely the same model to answerquestions as is used for the SAE features.

3. How well does the above feature generalise to hallucination detection?
- We can try to detect hallucinations outside medical contexts, not just in diseases.
