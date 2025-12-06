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
    Welcome to the **Alzheimer Disease Prediction Application**. This interactive tool is designed 
    to help both technical and non-technical users understand, analyze, and predict Alzheimer's 
    disease using machine learning models.
    
    This application prioritizes **high sensitivity (recall)** to minimize false negatives, as 
    missing true cases of Alzheimer's disease has significant clinical consequences.
    """)
    
    st.header("📋 Application Overview")
    st.markdown("""
    This application consists of **6 main pages**, each serving a specific purpose:
    """)
    
    # Page descriptions
    pages_info = [
        {
            "page": "1. Introduction",
            "icon": "📖",
            "description": "You are here! This page provides an overview of the application and explains what each page does."
        },
        {
            "page": "2. Technical Description",
            "icon": "🔬",
            "description": "Detailed technical documentation about the project, including implementation decisions, model descriptions, and evaluation strategies. This page automatically loads content from the project summary file."
        },
        {
            "page": "3. Statistical Analysis",
            "icon": "📊",
            "description": "Interactive exploratory data analysis (EDA) page where you can:\n- Upload your own dataset or use the default dataset\n- Apply filters and thresholds to explore specific data ranges\n- View statistical summaries, correlations, and visualizations\n- Download filtered datasets for further analysis"
        },
        {
            "page": "4. Model Training",
            "icon": "🤖",
            "description": "Train and evaluate machine learning models:\n- Select from 4 different model types (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)\n- Upload custom datasets or use the default dataset\n- View comprehensive performance metrics and visualizations\n- Perform hyperparameter tuning\n- Compare model performance"
        },
        {
            "page": "5. Explainability",
            "icon": "🔍",
            "description": "Understand model predictions using explainability techniques:\n- Select from multiple explainability methods (SHAP, ELI5, LIME)\n- View global feature importance across the entire model\n- Analyze local explanations for individual predictions\n- Compare insights across different explainability methods"
        },
        {
            "page": "6. Prediction",
            "icon": "🎯",
            "description": "Make predictions for individual patients:\n- Input patient features manually or use default values\n- Get instant predictions with probability scores\n- View which features contribute most to the prediction\n- Understand the model's reasoning for each case"
        }
    ]
    
    for info in pages_info:
        with st.expander(f"{info['icon']} **{info['page']}**", expanded=True):
            st.markdown(info['description'])
    
    st.header("🎯 Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📈 Interactive Analysis**
        - Upload and filter datasets
        - Real-time visualizations
        - Customizable thresholds
        """)
    
    with col2:
        st.markdown("""
        **🤖 Multiple Models**
        - 4 different ML algorithms
        - Hyperparameter tuning
        - Performance comparison
        """)
    
    with col3:
        st.markdown("""
        **🔍 Explainability**
        - SHAP, ELI5, LIME
        - Global and local insights
        - Clinical interpretability
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
    1. **New Users**: Start with this Introduction page to understand the application structure.
    
    2. **Data Analysis**: Navigate to **Statistical Analysis** to explore the dataset and understand 
    the features.
    
    3. **Model Training**: Go to **Model Training** to train and evaluate models on your data.
    
    4. **Understanding Predictions**: Use **Explainability** to understand how models make decisions.
    
    5. **Making Predictions**: Use **Prediction** to get predictions for individual patients.
    """)
    
    st.markdown("---")
    st.markdown("**Ready to get started?** Use the sidebar to navigate to any page!")

