# # EXAMPLE USAGE

# model_path = "../assets/hallucination_model.pkl"
# api_key ='sk-goodfire-9IJgLomji2zNdvFLPsTYPQvPPr_kUC19bFTh0HgT9h6SikyfPB7WmQ'

# # Get the classifier model
# import src.classifier.hallucination_classifier
# classifier = HallucinationClassifier(
#     model_path=model_path,
#     api_key=api_key
# )

# # Prepare example data
# prompt_example = """You are a medical expert and this is a multiple choice exam question. Please respond with the integer index of the CORRECT answer only; [0,1,2,3].
# Mental disorders in the Diagnostic and Statistical Manual of the American Psychiatric Association, following a personality disorder that belongs to the species (cluster) with three other different?
# Options: {"0": "Borderline", "1": "Antisocial", "2": "Paranoid", "3": "Drama type"}
# """

# # Get Prediction
# # Note usage of debug=True to demonstrate feature activations
# # In production you may want to use debug = False

# prediction, confidence = classifier.predict(prompt_example, debug=True)

# print(f"Prediction: {'Hallucinated' if prediction == 1 else 'Truthful'}")
# print(f"Confidence: {confidence:.2f}")

import pickle
import goodfire
from typing import List, Dict, Tuple, Any
import numpy as np

class HallucinationClassifier:
    def __init__(self, model_path: str, api_key: str):
        """
        Initialize the hallucination classifier with a saved model and features.
        
        Args:
            model_path: Path to the saved pickle file containing both the model and features
            api_key: Goodfire API key for accessing the service
        """
        # Load the model and features
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.features = model_data['features']
        self.client = goodfire.Client(api_key)
        self.variant = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")

    def _format_prompt(self, question: str) -> List[Dict[str, str]]:
        """Format a question into the expected prompt structure."""
        introduction = ("You are a medical expert and this is a multiple choice exam question. "
                      "Please respond with the integer index of the CORRECT answer only; [0,1,2,3].")
        return [{"role": "user", "content": f"{introduction}\n\n{question}"}]

    def _get_feature_activations(self, prompt: List[Dict[str, str]]) -> List[float]:
        """Get feature activations for the input prompt."""
        context = self.client.features.inspect(
            prompt,
            model=self.variant,
            features=self.features
        )
        
        # Get activations for our specific features
        activations = []
        features_dict = {f.uuid: 0.0 for f in self.features}
        
        for feature_act in context.top(k=len(self.features)):
            if feature_act.feature.uuid in features_dict:
                features_dict[feature_act.feature.uuid] = feature_act.activation
        
        # Maintain order matching the original features
        for feature in self.features:
            activations.append(features_dict[feature.uuid])
            
        return activations

    def predict(self, question: str, debug: bool = False) -> Tuple[int, float]:
        """
        Predict whether a given question-answer pair is likely to contain hallucination.
        
        Args:
            question: The question text
            debug: If True, print debugging information about feature activations
            
        Returns:
            Tuple containing:
            - Prediction (0 for truthful, 1 for hallucinated)
            - Confidence score (probability of the predicted class)
        """
        # Format the prompt
        prompt = self._format_prompt(question)
        
        # Get feature activations
        activations = self._get_feature_activations(prompt)
        
        if debug:
            print("\nFeature Activations:")
            for feature, activation in zip(self.features, activations):
                print(f"{feature.label}: {activation:.4f}")
            
            # Get the decision path
            decision_path = self.model.decision_path([activations])
            feature_importance = self.model.feature_importances_
            
            print("\nFeature Importance in Model:")
            for feature, importance in zip(self.features, feature_importance):
                print(f"{feature.label}: {importance:.4f}")
            
            print("\nDecision Path:")
            node_indicator = decision_path[0]
            leaf_id = self.model.apply([activations])[0]
            
            # Get thresholds and feature indices for each node in path
            for node_id in node_indicator.indices:
                if node_id != leaf_id:
                    feature_idx = self.model.tree_.feature[node_id]
                    threshold = self.model.tree_.threshold[node_id]
                    feature_name = self.features[feature_idx].label
                    feature_value = activations[feature_idx]
                    print(f"Node {node_id}: {feature_name} = {feature_value:.4f} {'<=' if feature_value <= threshold else '>'} {threshold:.4f}")
        
        # Make prediction
        prediction = self.model.predict([activations])[0]
        probabilities = self.model.predict_proba([activations])[0]
        confidence = probabilities[prediction]
        
        if debug:
            print(f"\nProbabilities:")
            print(f"Truthful: {probabilities[0]:.4f}")
            print(f"Hallucinated: {probabilities[1]:.4f}")
        
        return int(prediction), float(confidence)
