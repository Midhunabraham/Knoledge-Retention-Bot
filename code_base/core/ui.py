import streamlit as st

def render_header():
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("assets/logo.png", width=90)
    with col2:
        st.title("Knowledge Retention & Training Bot")
        st.caption("AI-powered knowledge assistant")
