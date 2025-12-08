"""
Page 1: Introduction
Explains the purpose of the application and each page
"""

import streamlit as st


def show():
    # Hero section with title and key message
    col_title, col_icon = st.columns([4, 1])
    with col_title:
        st.title("Alzheimer Disease Prediction Application")
        st.markdown("### Clinical Decision Support Tool")
    with col_icon:
        st.markdown("<div style='text-align: center; font-size: 60px;'>🧠</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Welcome message in a highlighted box
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; margin-bottom: 20px;'>
    <h3 style='color: #1f77b4; margin-top: 0;'>Welcome</h3>
    <p style='font-size: 16px; line-height: 1.6;'>
    This clinical decision support tool assists healthcare providers in assessing patient risk for 
    Alzheimer's disease using advanced machine learning models. The application is optimized to 
    <strong>minimize missed cases</strong> by prioritizing high sensitivity, ensuring patients who need 
    early intervention are not overlooked.
    </p>
    <p style='font-size: 14px; color: #666; margin-bottom: 0;'>
    <strong>Note:</strong> This tool supports clinical decision-making and should be used in conjunction 
    with comprehensive clinical assessment.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Application Overview with cards
    st.header("Application Overview")
    st.markdown("""
    <p style='font-size: 16px; color: #666; margin-bottom: 30px;'>
    Navigate through the application using the sections below. Each page serves a specific purpose 
    in the clinical workflow.
    </p>
    """, unsafe_allow_html=True)
    
    # Page descriptions with navigation buttons - using cards
    pages_info = [
        {
            "page": "Statistical Analysis",
            "icon": "📊",
            "short_desc": "Explore patient data and identify risk factor patterns",
            "description": "Review patient demographics, clinical measurements, and analyze relationships between risk factors and Alzheimer's diagnosis. All visualizations include clinical interpretation guides.",
            "features": ["Patient demographics review", "Risk factor analysis", "Statistical summaries", "Data filtering"],
            "nav_page": "📊 Statistical Analysis"
        },
        {
            "page": "Model Training",
            "icon": "⚙️",
            "short_desc": "Train and evaluate prediction models with clinical metrics",
            "description": "Train clinically-validated prediction models and view performance metrics with clinical interpretation. Understand model accuracy, sensitivity, and specificity.",
            "features": ["Model selection", "Performance metrics", "ROC & confusion matrix visualizations", "Threshold optimization"],
            "nav_page": "⚙️ Model Training"
        },
        {
            "page": "Prediction",
            "icon": "🎯",
            "short_desc": "Assess individual patient risk",
            "description": "Enter patient clinical information to get risk assessments with probability scores and understand the clinical reasoning behind predictions.",
            "features": ["Patient data input", "Risk assessment", "Probability scores", "Factor contributions"],
            "nav_page": "🎯 Prediction"
        },
        {
            "page": "Explainability",
            "icon": "🔍",
            "short_desc": "Understand how the model makes predictions",
            "description": "See which clinical factors are most important for predictions and understand why specific patients are flagged as high risk.",
            "features": ["Feature importance analysis", "Individual prediction explanations", "SHAP and LIME methods", "Clinical context"],
            "nav_page": "🔍 Explainability"
        }
    ]
    
    # Display pages in a grid layout (2 columns)
    for i in range(0, len(pages_info), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(pages_info):
                info = pages_info[i + j]
                with col:
                    # Create a card-like container
                    st.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; 
                                border: 1px solid #dee2e6; margin-bottom: 20px; height: 100%;'>
                        <div style='font-size: 32px; margin-bottom: 10px;'>{info['icon']}</div>
                        <h3 style='color: #1f77b4; margin-top: 0;'>{info['page']}</h3>
                        <p style='color: #666; font-size: 14px; margin-bottom: 15px;'>{info['short_desc']}</p>
                        <ul style='font-size: 13px; color: #555; padding-left: 20px;'>
                    """, unsafe_allow_html=True)
                    
                    for feature in info['features']:
                        st.markdown(f"<li>{feature}</li>", unsafe_allow_html=True)
                    
                    st.markdown("</ul></div>", unsafe_allow_html=True)
                    
                    # Navigation button
                    if st.button(f"Go to {info['page']}", key=f"nav_{info['nav_page']}", use_container_width=True):
                        st.session_state.current_page = info['nav_page']
                        st.rerun()
    
    st.markdown("---")
    
    # Key Features section
    st.header("Key Features")
    st.markdown("""
    <p style='font-size: 16px; color: #666; margin-bottom: 20px;'>
    The application is designed with clinicians in mind, featuring intuitive interfaces and 
    comprehensive interpretation guides for all visualizations and metrics.
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;'>
        <h4 style='color: #856404; margin-top: 0;'>📈 Data Exploration</h4>
        <ul style='font-size: 14px; color: #555;'>
        <li>Review patient demographics</li>
        <li>Identify risk factor patterns</li>
        <li>Filter by clinical criteria</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #0dcaf0;'>
        <h4 style='color: #055160; margin-top: 0;'>⚙️ Clinical Models</h4>
        <ul style='font-size: 14px; color: #555;'>
        <li>Validated prediction algorithms</li>
        <li>Performance metrics with interpretation</li>
        <li>Optimized for high sensitivity</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background-color: #d4edda; padding: 15px; border-radius: 8px; border-left: 4px solid #198754;'>
        <h4 style='color: #0f5132; margin-top: 0;'>🔍 Transparent Predictions</h4>
        <ul style='font-size: 14px; color: #555;'>
        <li>Understand prediction factors</li>
        <li>See clinical reasoning</li>
        <li>Interpretable assessments</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Important Notes in a warning box
    st.markdown("""
    <div style='background-color: #fff3cd; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107;'>
    <h3 style='color: #856404; margin-top: 0;'>⚠️ Important Notes</h3>
    <ul style='font-size: 15px; line-height: 1.8; color: #856404;'>
    <li><strong>Clinical Context:</strong> This tool is designed for research and educational purposes. 
    Always consult with medical professionals for actual clinical decisions.</li>
    <li><strong>Data Privacy:</strong> Ensure patient data is properly anonymized before uploading.</li>
    <li><strong>Model Limitations:</strong> Models are trained on specific datasets and may not generalize 
    to all populations. External validation is recommended before clinical deployment.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Getting Started section with visual workflow
    st.header("Getting Started")
    st.markdown("""
    <p style='font-size: 16px; color: #666; margin-bottom: 20px;'>
    Follow this recommended workflow to get the most out of the application:
    </p>
    """, unsafe_allow_html=True)
    
    # Visual workflow steps
    workflow_steps = [
        {"step": "1", "title": "Start Here", "desc": "Review this introduction to understand the tool's capabilities", "icon": "📍"},
        {"step": "2", "title": "Explore Data", "desc": "Use Statistical Analysis to understand patient characteristics", "icon": "📊"},
        {"step": "3", "title": "Train Model", "desc": "Train a prediction model and review performance metrics", "icon": "⚙️"},
        {"step": "4", "title": "Assess Patients", "desc": "Enter patient information to get risk assessments", "icon": "🎯"},
        {"step": "5", "title": "Understand Model", "desc": "See which clinical factors the model considers important", "icon": "🔍"}
    ]
    
    # Display workflow in a horizontal layout
    workflow_cols = st.columns(5)
    for idx, (col, step) in enumerate(zip(workflow_cols, workflow_steps)):
        with col:
            st.markdown(f"""
            <div style='text-align: center; padding: 15px; background-color: #f8f9fa; 
                        border-radius: 10px; border: 2px solid #1f77b4; height: 100%;'>
                <div style='font-size: 24px; margin-bottom: 10px;'>{step['icon']}</div>
                <div style='font-weight: bold; color: #1f77b4; font-size: 14px; margin-bottom: 8px;'>
                    Step {step['step']}
                </div>
                <div style='font-weight: bold; color: #333; font-size: 13px; margin-bottom: 5px;'>
                    {step['title']}
                </div>
                <div style='font-size: 11px; color: #666; line-height: 1.4;'>
                    {step['desc']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if idx < 4:
                st.markdown("<div style='text-align: center; font-size: 20px; color: #1f77b4; margin-top: -20px;'>→</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tip box
    st.markdown("""
    <div style='background-color: #e7f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #0d6efd; margin-top: 20px;'>
    <strong>💡 Tip:</strong> All visualizations include interpretation guides to help you understand 
    what the charts and graphs mean in clinical terms. Look for the "How to Interpret" expandable 
    sections throughout the application.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Call to action
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>
        <h3 style='color: #1f77b4; margin-bottom: 10px;'>Ready to get started?</h3>
        <p style='font-size: 16px; color: #666; margin-bottom: 0;'>
        Use the navigation buttons above or the sidebar menu to explore the application.
        </p>
    </div>
    """, unsafe_allow_html=True)

