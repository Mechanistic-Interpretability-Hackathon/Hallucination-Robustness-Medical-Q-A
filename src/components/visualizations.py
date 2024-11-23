import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def render_visualizations(activated_features):
    st.markdown("---")
    st.header("Visualizations")
    cols = st.columns([1, 1])

    with cols[0]:
        st.subheader("Hallucination Probability")
        x = np.linspace(0, 1, 10)
        y = np.sin(x)  # Example data
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title("Hallucination Probability vs. Feature Weight")
        ax.set_xlabel("Feature Weight")
        ax.set_ylabel("Probability")
        st.pyplot(fig)

    with cols[1]:
        usefulness = [0.7, 0.4, 0.6]  # Example usefulness values
        st.subheader("Feature Usefulness")
        fig, ax = plt.subplots()
        ax.bar(activated_features, usefulness)
        ax.set_title("Feature Usefulness")
        ax.set_xlabel("Features")
        ax.set_ylabel("Usefulness Score")
        st.pyplot(fig)
