# EXAMPLE USAGE FOR SVM MODEL

# model_path = "../classifier/hallucination_classifier_svm.pkl"

# prompt_example = "this is my prompt"

# classifier = SVMHallucinationClassifier(
#     model_path=model_path,
#     api_key=api_key
# )

# # get prediction
# # prediction = 1 indicates hallucinated
# # prediction = 0 indicates truthful
# prediction, confidence = classifier.predict(prompt_example, debug=True)


import pickle
import goodfire
from typing import List, Dict, Tuple, Any
import numpy as np
import sklearn

class SVMHallucinationClassifier:
    def __init__(self, model_path: str, api_key: str):
        """
        Initialize the hallucination classifier with a saved SVM model and features.
        
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
            
            # For SVM, we can show feature importance through the absolute values of coefficients
            # Note: This only works for linear SVM. For non-linear kernels, feature importance
            # cannot be directly computed from the model coefficients
            if hasattr(self.model, 'coef_'):
                print("\nFeature Importance in Model (based on absolute coefficient values):")
                feature_importance = np.abs(self.model.coef_[0])
                for feature, importance in zip(self.features, feature_importance):
                    print(f"{feature.label}: {importance:.4f}")
            
            # For SVM, we can show the distance from the decision boundary
            decision_function = self.model.decision_function([activations])[0]
            print(f"\nDistance from decision boundary: {decision_function:.4f}")
            
        # Make prediction
        prediction = self.model.predict([activations])[0]
        probabilities = self.model.predict_proba([activations])[0]
        confidence = probabilities[prediction]
        
        if debug:
            print(f"\nProbabilities:")
            print(f"Truthful: {probabilities[0]:.4f}")
            print(f"Hallucinated: {probabilities[1]:.4f}")
        
        return int(prediction), float(confidence)