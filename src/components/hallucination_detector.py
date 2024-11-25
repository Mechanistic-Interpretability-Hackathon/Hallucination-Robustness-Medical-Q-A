import streamlit as st
import os
from src.classifier.hallucination_classifier import HallucinationClassifier
from config.config import gf_api_key

def render_hallucination_detector(prompt):
  model_path = "src/classifier/hallucination_classifier_svm.pkl"
  st.markdown('--')
  st.header("Hallucination Detection")
  
  classifier = HallucinationClassifier(model_path=model_path, api_key=gf_api_key)
  prediction, confidence = classifier.predict(prompt, debug=False)
  result = "Hallucinated" if prediction == 1 else "Truthful"
  st.success(f"Prediction: {result}") if result == "Truthful" else st.warning(f"Prediction: {result}")
  st.info(f"I am: {confidence * 100:.2f}% confident")
  
