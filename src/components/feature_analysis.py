import streamlit as st
import matplotlib.pyplot as plt

def render_feature_analysis(activated_features):
    st.markdown("---")
    st.header("Feature Steering and Analysis")
    cols = st.columns([2, 3])  # Layout

    with cols[0]:
        st.subheader("Feature Steering")
        sliders = {}
        for feature in activated_features:
            sliders[feature] = st.slider(f"Adjust {feature}:", min_value=0.0, max_value=1.0, step=0.1, value=0.5)

    with cols[1]:
        st.subheader("Activated Features")
        fig, ax = plt.subplots()
        ax.bar(activated_features, [sliders[f] for f in activated_features])
        ax.set_title("Feature Adjustments")
        ax.set_xlabel("Features")
        ax.set_ylabel("Adjusted Values")
        st.pyplot(fig)
