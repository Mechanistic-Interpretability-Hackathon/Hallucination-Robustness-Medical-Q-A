import streamlit as st

def render_navbar():
    # st.sidebar.image("src/assets/Gradients Anatomy_logo1.webp", use_container_width=True)
    st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        padding: 0;
    }
    .nav-link {
        font-size: 16px;
        color: #333;
        text-decoration: none;
        padding: 10px 15px;
        display: flex;
        align-items: center;
        border-radius: 5px;
    }
    .nav-link:hover {
        background-color: #f0f0f0;
        color: #000;
    }
    .nav-link-active {
        background-color: #e6f7ff;
        color: #007bff;
        font-weight: bold;
    }
    .icon {
        margin-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)  
    st.sidebar.markdown(
        """
        <div>
            <a class="nav-link nav-link-active" href="#home">
                <span class="icon">🏠</span> Home
            </a>
            <a class="nav-link" href="#diagnostics">
                <span class="icon">📊</span> Diagnostics
            </a>
            <a class="nav-link" href="#feature-analysis">
                <span class="icon">🔬</span> Feature Analysis
            </a>
            <a class="nav-link" href="#about">
                <span class="icon">ℹ️</span> About
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
