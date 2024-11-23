import streamlit as st
from components.navbar import render_navbar
from components.footer import render_footer
from components.feature_analysis import render_feature_analysis
from components.visualizations import render_visualizations
from utils.mock_data import mock_responses, activated_features


st.set_page_config(page_title="Explainable Diagnostic Assistant", layout="wide")

render_navbar()

st.title("Diagnostics Page")
st.markdown("Refine model outputs using feature steering and analyze response quality.")
st.markdown("---")


st.header("Model Inference")
mock_responses_keys = list(mock_responses.keys())

cols = st.columns([3, 1])  
with cols[0]:
    selected_question = st.selectbox("Select a predefined question:", mock_responses_keys)
    if st.button("Submit Query"):
        response = mock_responses.get(selected_question, "I'm sorry, I don't have an answer for that.")
        st.session_state['response'] = response
        st.write(st.session_state['response'])


render_feature_analysis(activated_features)

render_visualizations(activated_features)

render_footer()






