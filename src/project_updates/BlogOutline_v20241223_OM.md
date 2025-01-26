==========================
# INSTRUCTIONS FROM APART REF BLOG
==========================

[https://apartresearch.notion.site/Blogpost-Studio-Deliverable-127fcfd1de9d80548632e4b2ad3af946]

==========================
# CRITERIA FOR HACKATHON
==========================

Reproduced here to help guide plan for the blog.

## 1. "Advancing Interpretability"

- Does the project contribute to the field of mechanistic interpretability? 
- Does it provide new insights into understanding or steering AI model behavior? 
- How well does it move us towards reprogramming AI models? How original and innovative is the approach?

## 2. "Relevance for AI Safety"

- Possible questions to answer in this criteria: How important is the contribution to advancing the field of AI safety? 
- Do we expect the results to generalize beyond the specific case(s) presented in the submission? 
- Does the approach introduce new safety mechanisms or enhance existing ones in innovative ways?

## 3. "Methodology & Presentation

- How well is the project executed from a technical standpoint?
- Is the code well-structured, documented, and reproducible?
- How effectively does it utilize Goodfire's SDK/API and other provided resources?
- How clearly and effectively is the research presented in the paper?
- Quality of visualizations and demos (if applicable), Clarity of methodology explanation and results interpretation))

==========================
# PROPOSED BLOG OUTLINE
==========================

# Detecting LLM Knowledge Awareness in a Medical Context

Language model hallucination poses clear risks in medical applications. Yet until recently, we've lacked reliable methods to detect when models are operating beyond their knowledge boundaries. Recent research, Ferrando et al.'s paper "Do I know this entity.." [1] suggests that models develop internal representations of their own knowledge through training, which they refer to as 'knowledge awareness'. Their work was completed using the LLM 'Gemma 2-2B' and GemmaScope sparse auto-encoders (SAE). It analyzed Gemma's internal representations, discovering that specific directions [2] in the model's representation space correlate with whether it recognizes entities.

[**Perhaps add a short subsection illustrating “high-stakes hallucination” scenarios in medicine? e.g., wrong dosage advice or fictitious clinical trials, to underscore the necessity of robust detection methods.**]

Ferrando et al "demonstrate these directions causally affect knowledge refusal in the chat model, i.e. by steering with these directions, we can cause the model to hallucinate rather than refuse on unknown entities, and refuse to answer questions about known entities". In doing so, they "go beyond merely understanding knowledge refusal, and find SAE latents, seemingly representing uncertainty, that are predictive of incorrect answers."

Simultaneously, our team tackled the related challenge of using a classifier to identify hallucinations in Llama-3.3 models but using a production-grade tool, namely Goodfire's SAE via API to quickly produce the SAE activations for the classifier. We found similar patterns, SAE features which indicate hallucination, but with specific focus on medical knowledge representation.

Both our and Ferrando's approaches converge on a crucial insight: The model training process, which directs the model to learn how best to present the information acquired during pre-training, may imbue it with internal mechanisms for assessing the limits of its own knowledge boundaries.

In Ferrando et al's paper, this manifested as SAE activation patterns that distinctly separated known from unknown entities. In our work, we found similar patterns specific to medical knowledge, including a feature, possibly arising from RLHF, that activates when the model has sufficient information to avoid hallucination: "The model should not recommend technological or medical interventions".

[**Consider adding a small figure or table comparing “Gemma entity directions” to “Goodfire medical features” side-by-side, illustrating how both approaches differentiate known vs. unknown. Also, briefly mention how bridging the two approaches strengthens interpretability claims.**]

## Methods Comparison 

Ferrando et al. demonstrated their approach by examining entity recognition, using GemmaScope to analyze how Gemma processes known entities like 'Michael Jordan' versus unknown entities like 'Michael Joordan'. They found specific directions in the model's representation space that activate differently for known versus unknown entities.

Our initial approach used Goodfire's SAE API to explore similar patterns in medical knowledge, developing classification models to identify hallucinations, trained using the MedHALT labelled medical hallucination dataset [3]. The classification models, a decision tree and a support vector machine, then indicate the most important features for medical hallucination. 

These features were used to steer the model to lower levels of hallucination and increased accuracy. The work demonstrated that production-grade tools could potentially be used for this type of analysis. However, it's crucial to note that our findings represent an investigation of such tools rather than a system ready for real medical applications.

[**Expand with visualizations comparing activation patterns between the two approaches**]

[**Add a short paragraph clarifying how Ferrando’s “final-token-of-entity” analysis compares to Goodfire’s “steered features.” Summarize how adopting their token-based approach may yield consistent or domain-specific interpretability insights in medical Q&A.**]

## Extending Our Analysis

We have extended our research to align more directly with Ferrando et al.'s methodology. Following their approach, we have categorized medical entities as 'known' or 'unknown' based on the model's ability to correctly recall their attributes. This systematic approach will allow us to directly compare our findings with theirs, potentially validating whether the knowledge-awareness mechanisms they discovered in Gemma have analogues in other models and domains.

By accessing Goodfire's SAE activations for selected known and unknown medical entities, we aim to identify whether similar distinctive activation differences emerge for medical knowledge.

1. A decision tree classifier using three key features, providing interpretability at the cost of precision
2. An SVM classifier using 53 features, offering better discrimination but less interpretability

[**Add performance tables or charts (precision, recall, F1) for each classifier to illustrate the trade-offs. Note how steering might over-warn or under-warn in real deployment contexts.**]

## Feature Analysis and Causal Effects

While our initial work identified promising features through training classifiers via SAE activitions from Goodfire's API, particularly 'The model should not recommend technological or medical interventions', this extended work allows us to examine whether these features align with those arising from the entity-recognition directions approach employed in Ferrando et al. 

This comparison was intended to reveal whether:
- (a) knowledge-awareness mechanisms are consistent across these two approaches
- (b) whether the entity-recognition approach demonstrated by Ferrando et al using GemmaScope SAE's on Gemma 2-2B are effective using Goodfire SAE's on Llama-3.3 models.

[**Incorporate a short bullet list or figure demonstrating how “steering” these features affects hallucination vs. refusal rates, reinforcing the causal relevance of the directions.**]

## Future Directions

By extending our analysis to match Ferrando et al.'s methodology, we hope to have contributed to a broader understanding of knowledge-awareness in language models. This alignment of research methods could help establish whether these mechanisms are fundamental to language models or specific to certain architectures or domains.

Future work may include:

- Detailed comparison of activation patterns across different medical entity types
- Analysis of how these mechanisms scale with model size
- Investigation of potential universal measures of model knowledge confidence
- Exploration of interaction with other safety-critical behaviors
- Examination of cross-domain generalization
- Performance comparisons between research and production-grade tools

[**Add short mention of regulatory or ethical considerations for medical deployment (e.g., how disclaimers or human-in-the-loop oversight might be integrated).**]

## Conclusion

The convergence of research insights from Ferrando et al. and our production-focused exploration suggests promising directions for understanding and potentially controlling model knowledge awareness. While significant work remains before such systems could be safely deployed in critical domains like medicine, these findings advance our understanding of how models represent and assess their own knowledge boundaries.

[**We are mostly concerned with use of Goodfire to identify hallucinations, not steer a hallucinating model. Nevertheless, consider a note on potential negative side effects of steering (e.g., missing important info by over-refusing). This acknowledges real-world usability vs. safety trade-offs.**]


[1] Ferrando, J., Obeso, O., Rajamanoharan, S., & Nanda, N. (2024). Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models. arXiv preprint arXiv:2411.14257.
[2] A 'direction' is a combination of activations of an SAE, which is represented as a vector, i.e. a direction.
[3] Pal et al, 2023, Med-HALT: Medical Domain Hallucination Test for LLMs. arXiv preprint arxiv:2307.15343

[**Other papers related to Ferrando et al: https://www.connectedpapers.com/main/ee480ff85412144887ab8f48eef37db273d9d952/Do-I-Know-This-Entity%3F-Knowledge-Awareness-and-Hallucinations-in-Language-Models/graph**]

NOTE: Other Hackathon paper on a similar path:
[https://www.apartresearch.com/project/unveiling-latent-beliefs-using-sparse-autoencoders]

