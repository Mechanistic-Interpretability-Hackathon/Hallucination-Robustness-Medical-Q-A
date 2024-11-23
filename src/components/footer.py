import streamlit as st

def render_footer():
    st.markdown(
        """
        <style>
        .full-width-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #333333; /* Dark gray */
            color: #f0f0f0;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            z-index: 9999;
        }
        .full-width-footer a {
            color: #1e90ff;
            text-decoration: none;
        }
        .full-width-footer a:hover {
            text-decoration: underline;
        }
        </style>
        <div class="full-width-footer">
            Developed by <a href="https://yourportfolio.com" target="_blank">Gradients Anatomy</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
