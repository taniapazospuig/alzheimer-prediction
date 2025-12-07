"""
Page 1: Introduction
Explains the purpose of the application and each page
"""

import streamlit as st


def show():
    st.title("🧠 Alzheimer Disease Prediction Application")
    st.markdown("---")
    
    st.header("Welcome")
    st.markdown("""
    Welcome to the **Alzheimer Disease Prediction Application**. This clinical decision support tool 
    is designed to assist healthcare providers in assessing patient risk for Alzheimer's disease 
    using advanced machine learning models.
    
    **Clinical Priority**: This application is optimized to **minimize missed cases** (false negatives). 
    We prioritize identifying patients who may have Alzheimer's disease, even if this means some 
    patients without the disease are flagged for further evaluation. This approach ensures that 
    patients who need early intervention are not overlooked.
    
    **Important**: This tool is intended to support clinical decision-making, not replace it. 
    All predictions should be interpreted in conjunction with comprehensive clinical assessment.
    """)
    
    st.header("📋 Application Overview")
    st.markdown("""
    This application consists of **5 main pages**, each serving a specific purpose:
    """)
    
    # Page descriptions with navigation buttons
    pages_info = [
        {
            "page": "1. Introduction",
            "icon": "📖",
            "description": "You are here! This page provides an overview of the application and explains what each page does.",
            "nav_page": None
        },
        {
            "page": "2. Statistical Analysis",
            "icon": "📊",
            "description": "Explore patient data and identify patterns:\n- Review patient demographics and clinical measurements\n- Analyze relationships between risk factors and Alzheimer's diagnosis\n- View statistical summaries with clinical interpretation guides\n- Filter data by specific criteria (e.g., age range, MMSE scores)",
            "nav_page": "📊 Statistical Analysis"
        },
        {
            "page": "3. Model Training",
            "icon": "⚙️",
            "description": "Train and evaluate prediction models:\n- Select from clinically-validated model types\n- View model performance metrics with clinical interpretation\n- Understand model accuracy, sensitivity, and specificity\n- See visualizations (ROC curves, confusion matrices) with interpretation guides",
            "nav_page": "⚙️ Model Training"
        },
        {
            "page": "4. Explainability",
            "icon": "🔍",
            "description": "Understand how the model makes predictions:\n- See which clinical factors are most important for predictions\n- Understand why specific patients are flagged as high risk\n- View feature importance with clinical context\n- Get explanations for individual patient predictions",
            "nav_page": "🔍 Explainability"
        },
        {
            "page": "5. Prediction",
            "icon": "🎯",
            "description": "Assess individual patient risk:\n- Enter patient clinical information\n- Get risk assessment with probability scores\n- See which factors contribute most to the risk assessment\n- Understand the clinical reasoning behind the prediction",
            "nav_page": "🎯 Prediction"
        }
    ]
    
    for info in pages_info:
        with st.expander(f"{info['icon']} **{info['page']}**", expanded=True):
            st.markdown(info['description'])
            if info['nav_page']:
                if st.button(f"Go to {info['page']}", key=f"nav_{info['nav_page']}"):
                    st.session_state.current_page = info['nav_page']
                    st.rerun()
    
    st.header("🎯 Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📈 Data Exploration**
        - Review patient demographics and clinical data
        - Identify risk factor patterns
        - Filter by clinical criteria
        """)
    
    with col2:
        st.markdown("""
        **⚙️ Clinical Models**
        - Validated prediction algorithms
        - Performance metrics with clinical interpretation
        - Optimized for high sensitivity
        """)
    
    with col3:
        st.markdown("""
        **🔍 Transparent Predictions**
        - Understand which factors drive predictions
        - See clinical reasoning for each case
        - Interpretable risk assessments
        """)
    
    st.header("⚠️ Important Notes")
    st.info("""
    - **Clinical Context**: This tool is designed for research and educational purposes. 
    Always consult with medical professionals for actual clinical decisions.
    
    - **Data Privacy**: Ensure patient data is properly anonymized before uploading.
    
    - **Model Limitations**: Models are trained on specific datasets and may not generalize 
    to all populations. External validation is recommended before clinical deployment.
    """)
    
    st.header("🚀 Getting Started")
    st.markdown("""
    **Recommended Workflow for Clinicians:**
    
    1. **Start Here**: Review this introduction to understand the tool's capabilities and clinical focus.
    
    2. **Explore Data**: Use **Statistical Analysis** to understand patient characteristics and identify 
    risk factor patterns in your patient population.
    
    3. **Train Model**: Go to **Model Training** to train a prediction model. The tool will show you 
    how well the model performs, with clinical interpretation of all metrics and visualizations.
    
    4. **Understand Model**: Use **Explainability** to see which clinical factors the model considers 
    most important and understand its decision-making process.
    
    5. **Assess Patients**: Use **Prediction** to enter individual patient information and get 
    risk assessments with clear explanations.
    
    **💡 Tip**: All visualizations include interpretation guides to help you understand what the 
    charts and graphs mean in clinical terms.
    """)
    
    st.markdown("---")
    st.markdown("**Ready to get started?** Use the buttons in the Application Overview above or the sidebar to navigate to any page!")

