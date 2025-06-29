import streamlit as st

st.set_page_config(
    page_title="Test App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<style>body {background-color: #f0f2f6;}</style>", unsafe_allow_html=True)
st.title("Hello, Streamlit! Testing Deployment.")
st.write("If you see this, the basic app loads.")