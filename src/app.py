import streamlit as st
import logging
from components.navbar import render_navbar
from components.footer import render_footer
from components.feature_analysis import render_feature_analysis
from components.hallucination_detector import render_hallucination_detector
from components.visualizations import render_visualizations
from med_llm_evaluation.data_handler import DataHandler
from config.config import client, variant
from utils.utilities import split_prompt, activated_features


st.set_page_config(page_title="Explainable Diagnostic Assistant", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

render_navbar()

st.title("Diagnostics Page")
st.markdown("Refine model outputs using feature steering and analyze response quality.")
st.markdown("---")

st.header("Model Inference")

data_handler = DataHandler()
prompts, labels, subject_names = data_handler.load_data()
prompts_with_labels = dict(zip(prompts, labels))

system_prompt, user_prompts = split_prompt(prompts[:10])

cols = st.columns([3, 1])  
with cols[0]:
    selected_prompt = st.selectbox("Select a predefined question:", user_prompts)
    selected_user_prompt = system_prompt + "\n\n" + selected_prompt
    ground_truth_label = prompts_with_labels[selected_user_prompt]

    if st.button("Submit Query"):
        response_text = ""
        try: 
           response = client.chat.completions.create(
             messages=[{"role": 'user', "content": selected_user_prompt}],
             model = variant, 
             stream=False,
             max_completion_tokens=50,
           )
           response_text = response.choices[0].message['content']
        except Exception as e:
            st.write("**Error:**", e)

        if response_text:
            st.session_state['response'] = str(response_text)
            st.session_state['ground_truth_label'] = str(ground_truth_label)
            st.write("**Model Response:**", st.session_state['response'])
            st.write("**Ground truth**:", st.session_state['ground_truth_label'])
            st.write(f'**Model prediction**: Incorrect' if int(response_text) != int(ground_truth_label) else  '**Model Predicted**: Correct')
            render_hallucination_detector(selected_user_prompt)
        else:
            st.error("No response received from the model. Please try again.")


render_feature_analysis(activated_features)

render_visualizations(activated_features)

render_footer()






