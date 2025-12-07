"""
Alzheimer Disease Prediction - Streamlit Application
Main application file with multi-page structure
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Alzheimer Disease Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None  # Hide default menu
)

# Hide default Streamlit sidebar elements using CSS
st.markdown("""
<style>
    /* Hide Streamlit's default sidebar navigation if it appears */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    /* Style our custom sidebar */
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'data_dict' not in st.session_state:
    st.session_state.data_dict = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "📖 Introduction"

# Import pages
try:
    from pages import (
        page_1_introduction,
        page_3_statistical_analysis,
        page_4_model_training,
        page_5_explainability,
        page_6_prediction
    )
except ImportError as e:
    st.error(f"Error importing pages: {str(e)}")
    st.stop()

# Sidebar navigation - clean and minimal
st.sidebar.markdown("### Navigation Menu")
page_options = [
    "📖 Introduction",
    "📊 Statistical Analysis",
    "⚙️ Model Training",
    "🔍 Explainability",
    "🎯 Prediction"
]

# Get current page index for radio button
current_index = 0
if st.session_state.current_page in page_options:
    current_index = page_options.index(st.session_state.current_page)

page = st.sidebar.radio(
    "Select a page:",
    page_options,
    index=current_index,
    label_visibility="collapsed"
)

# Update session state when radio button changes
if page != st.session_state.current_page:
    st.session_state.current_page = page

# Route to selected page
if "📖 Introduction" in page or page == "1. Introduction":
    page_1_introduction.show()
elif "📊 Statistical Analysis" in page or page == "3. Statistical Analysis":
    page_3_statistical_analysis.show()
elif "⚙️ Model Training" in page or "Model Training" in page or page == "4. Model Training":
    page_4_model_training.show()
elif "🔍 Explainability" in page or page == "5. Explainability":
    page_5_explainability.show()
elif "🎯 Prediction" in page or page == "6. Prediction":
    page_6_prediction.show()

