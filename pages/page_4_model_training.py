"""
Page 4: Model Training
Train and evaluate machine learning models
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import logistic_regression, decision_tree, random_forest, gradient_boosting


def show():
    st.title("⚙️ Model Training")
    st.markdown("---")
    
    # Return to home button
    if st.button("🏠 Return to Home", type="secondary"):
        st.session_state.current_page = "📖 Introduction"
        st.rerun()
    
    st.markdown("---")
    
    # Clinical context
    st.info("""
    **Clinical Context**: This page trains prediction models to assess Alzheimer's disease risk. 
    All performance metrics and visualizations include clinical interpretation guides to help you 
    understand what they mean for patient care.
    """)
    
    # Sidebar for model selection and settings
    st.sidebar.header("⚙️ Model Configuration")
    
    # Model selection - Random Forest is the best performing model based on analysis
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
        index=2,  # Default to Random Forest (best performing model: 92.76% sensitivity, 93.38% precision)
        help="Random Forest is pre-selected as it provides the best performance (92.76% sensitivity, 93.38% precision, ROC-AUC: 0.9413). You can change to other models if needed."
    )
    
    # File upload
    st.sidebar.subheader("📁 Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file (or use default)",
        type=['csv'],
        help="Upload your dataset or use the default"
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Loaded: {data.shape}")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
            data = None
    else:
        try:
            data = pd.read_csv("alzheimers_disease_data.csv")
            st.sidebar.info(f"📊 Default dataset: {data.shape}")
        except Exception as e:
            st.sidebar.error(f"Error loading default: {str(e)}")
            data = None
    
    if data is None:
        st.error("Please upload a valid dataset or ensure the default dataset exists.")
        st.stop()
    
    # Hyperparameter tuning option
    use_tuning = st.sidebar.checkbox(
        "Enable Hyperparameter Tuning",
        value=False,
        help="This will take longer but may improve performance"
    )
    
    # Train button
    train_button = st.sidebar.button("🚀 Train Model", type="primary")
    
    # Main content area
    if train_button:
        with st.spinner("Training model... This may take a while."):
            try:
                # Prepare data based on model type
                if model_type == "Logistic Regression":
                    data_dict = logistic_regression.prepare_data(data)
                    results = logistic_regression.train_model(
                        data_dict, 
                        hyperparameter_tuning=use_tuning
                    )
                elif model_type == "Decision Tree":
                    data_dict = decision_tree.prepare_data(data)
                    results = decision_tree.train_model(data_dict)
                elif model_type == "Random Forest":
                    data_dict = random_forest.prepare_data(data)
                    results = random_forest.train_model(
                        data_dict,
                        hyperparameter_tuning=use_tuning
                    )
                elif model_type == "Gradient Boosting":
                    data_dict = gradient_boosting.prepare_data(data)
                    results = gradient_boosting.train_model(data_dict)
                
                # Store in session state
                st.session_state.trained_model = results['model']
                st.session_state.model_results = results
                st.session_state.data_dict = data_dict
                st.session_state.model_type = model_type
                
                st.success("✅ Model trained successfully!")
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                st.stop()
    
    # Display results if model is trained
    if st.session_state.trained_model is not None and st.session_state.model_results is not None:
        results = st.session_state.model_results
        model_type = st.session_state.get('model_type', 'Unknown')
        
        st.header("📊 Model Performance")
        
        # Clinical interpretation of metrics
        with st.expander("📖 Understanding Performance Metrics", expanded=False):
            st.markdown("""
            **Key Metrics for Clinical Decision-Making**:
            
            - **Accuracy**: Overall percentage of correct predictions. 
              *Clinical meaning*: How often the model is right overall.
            
            - **Sensitivity (Recall)**: Percentage of Alzheimer's patients correctly identified.
              *Clinical meaning*: How well the model catches true cases. **This is critical** - 
              we want to minimize missed cases. Target: ≥90%.
            
            - **Specificity**: Percentage of non-Alzheimer's patients correctly identified.
              *Clinical meaning*: How well the model avoids false alarms.
            
            - **Precision**: Of patients flagged as high risk, what percentage actually have Alzheimer's.
              *Clinical meaning*: When the model says "high risk," how often is it correct.
            
            - **F1-Score**: Balance between precision and sensitivity.
              *Clinical meaning*: Overall model performance considering both metrics.
            
            **For Alzheimer's screening, we prioritize high Sensitivity** to ensure we don't miss 
            patients who need early intervention, even if this means some false positives that 
            can be ruled out with further testing.
            """)
        
        # Metrics display
        metrics = results['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            accuracy = metrics['Accuracy']
            st.metric("Accuracy", f"{accuracy:.1%}", 
                     delta="Good" if accuracy > 0.80 else "Fair" if accuracy > 0.70 else "Needs Improvement")
        with col2:
            precision = metrics['Precision']
            st.metric("Precision", f"{precision:.1%}",
                     delta="Good" if precision > 0.70 else "Fair" if precision > 0.60 else "Needs Improvement")
        with col3:
            recall = metrics['Recall']
            st.metric("Sensitivity (Recall)", f"{recall:.1%}",
                     delta="Excellent" if recall >= 0.90 else "Good" if recall >= 0.80 else "Needs Improvement",
                     delta_color="normal" if recall >= 0.90 else "inverse")
        with col4:
            f1 = metrics['F1-Score']
            st.metric("F1-Score", f"{f1:.1%}",
                     delta="Good" if f1 > 0.70 else "Fair" if f1 > 0.60 else "Needs Improvement")
        
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            roc_auc = metrics.get('ROC-AUC', 0)
            st.metric("ROC-AUC", f"{roc_auc:.3f}",
                     delta="Excellent" if roc_auc > 0.90 else "Good" if roc_auc > 0.80 else "Fair")
        with col6:
            pr_auc = metrics.get('PR-AUC', 0)
            st.metric("PR-AUC", f"{pr_auc:.3f}",
                     delta="Excellent" if pr_auc > 0.80 else "Good" if pr_auc > 0.70 else "Fair")
        with col7:
            fn = metrics['False Negatives']
            st.metric("False Negatives (Missed Cases)", fn, 
                     delta=f"⚠️ {fn} missed" if fn > 0 else "✅ None",
                     delta_color="inverse" if fn > 0 else "normal")
        with col8:
            fp = metrics['False Positives']
            st.metric("False Positives", fp,
                     delta=f"⚠️ {fp} false alarms" if fp > 0 else "✅ None",
                     delta_color="inverse" if fp > 0 else "normal")
        
        # Detailed metrics table
        st.subheader("📋 Detailed Metrics")
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })
        st.dataframe(metrics_df, width='stretch')
        
        # Confusion Matrix
        st.subheader("📈 Visualizations")
        
        # Get plotting functions based on model type
        if model_type == "Logistic Regression":
            plot_cm = logistic_regression.plot_confusion_matrix
            plot_roc = logistic_regression.plot_roc_curve
            plot_pr = logistic_regression.plot_precision_recall_curve
            plot_importance = logistic_regression.plot_feature_importance
        elif model_type == "Decision Tree":
            plot_cm = decision_tree.plot_confusion_matrix
            plot_roc = decision_tree.plot_roc_curve
            plot_pr = decision_tree.plot_precision_recall_curve
            plot_importance = decision_tree.plot_feature_importance
        elif model_type == "Random Forest":
            plot_cm = random_forest.plot_confusion_matrix
            plot_roc = random_forest.plot_roc_curve
            plot_pr = random_forest.plot_precision_recall_curve
            plot_importance = random_forest.plot_feature_importance
        elif model_type == "Gradient Boosting":
            plot_cm = gradient_boosting.plot_confusion_matrix
            plot_roc = gradient_boosting.plot_roc_curve
            plot_pr = gradient_boosting.plot_precision_recall_curve
            plot_importance = gradient_boosting.plot_feature_importance
        
        # Plot confusion matrix and ROC curve side by side
        y_test = st.session_state.data_dict['y_test']
        y_pred = results['y_pred']
        y_proba = results['y_proba']
        
        col1, col2 = st.columns(2)
        with col1:
            fig_cm = plot_cm(y_test, y_pred, f"{model_type}")
            st.pyplot(fig_cm, width='stretch')
            
            # Confusion matrix interpretation
            with st.expander("📖 How to Interpret the Confusion Matrix", expanded=False):
                st.markdown("""
                **What this shows**: A breakdown of the model's predictions compared to actual diagnoses.
                
                **Understanding the Four Quadrants**:
                
                1. **Top Left (True Negatives)**: Patients correctly identified as NOT having Alzheimer's.
                   *Clinical meaning*: Correctly reassured patients. Higher is better.
                
                2. **Top Right (False Positives)**: Patients incorrectly flagged as having Alzheimer's.
                   *Clinical meaning*: False alarms - these patients will need further testing but don't have the disease.
                   Some false positives are acceptable to avoid missing true cases.
                
                3. **Bottom Left (False Negatives)**: Patients with Alzheimer's that the model missed.
                   *Clinical meaning*: **CRITICAL** - These are missed cases. We want this number as low as possible.
                   This is our highest priority to minimize.
                
                4. **Bottom Right (True Positives)**: Patients correctly identified as having Alzheimer's.
                   *Clinical meaning*: Correctly identified cases that need intervention. Higher is better.
                
                **Clinical Goal**: Minimize False Negatives (bottom left) even if this increases False Positives.
                """)
        
        with col2:
            fig_roc = plot_roc(y_test, y_proba, f"{model_type}")
            st.pyplot(fig_roc, width='stretch')
            
            # ROC curve interpretation
            with st.expander("📖 How to Interpret the ROC Curve", expanded=False):
                st.markdown("""
                **What this shows**: The model's ability to distinguish between patients with and without Alzheimer's.
                
                **Understanding the Curve**:
                - **X-axis (False Positive Rate)**: Percentage of non-Alzheimer's patients incorrectly flagged.
                - **Y-axis (True Positive Rate / Sensitivity)**: Percentage of Alzheimer's patients correctly identified.
                - **Diagonal line**: Performance of random guessing (no better than chance).
                
                **Clinical Interpretation**:
                - **Curve above diagonal**: Model is better than random guessing.
                - **Curve closer to top-left corner**: Better performance.
                  - High sensitivity (top) = catches most true cases
                  - Low false positive rate (left) = fewer false alarms
                - **AUC (Area Under Curve)**: 
                  - 0.90-1.0: Excellent discrimination
                  - 0.80-0.90: Good discrimination
                  - 0.70-0.80: Fair discrimination
                  - <0.70: Poor discrimination
                
                **For Alzheimer's screening**: We want high sensitivity (top of the curve), 
                even if this means more false positives (right side of curve).
                """)
        
        # Get feature importance before using it
        feature_importance = results['feature_importance']
        
        # Plot PR curve and feature importance side by side
        col3, col4 = st.columns(2)
        with col3:
            fig_pr = plot_pr(y_test, y_proba, f"{model_type}")
            st.pyplot(fig_pr, width='stretch')
            
            # Precision-Recall curve interpretation
            with st.expander("📖 How to Interpret the Precision-Recall Curve", expanded=False):
                st.markdown("""
                **What this shows**: The trade-off between precision and sensitivity, especially useful 
                when dealing with imbalanced datasets (more non-Alzheimer's than Alzheimer's patients).
                
                **Understanding the Curve**:
                - **X-axis (Recall/Sensitivity)**: Percentage of Alzheimer's patients correctly identified.
                - **Y-axis (Precision)**: Of patients flagged as high risk, what percentage actually have Alzheimer's.
                
                **Clinical Interpretation**:
                - **Curve closer to top-right**: Better performance (high precision AND high sensitivity).
                - **High Recall (right side)**: Catches most true cases - **this is our priority**.
                - **High Precision (top)**: When model flags someone, it's usually correct.
                - **AUC (Area Under Curve)**:
                  - 0.90-1.0: Excellent
                  - 0.80-0.90: Good
                  - 0.70-0.80: Fair
                  - <0.70: Poor
                
                **Clinical Goal**: We prioritize high recall (sensitivity) to minimize missed cases, 
                even if this means lower precision (more false positives that can be ruled out with further testing).
                """)
        
        with col4:
            fig_importance = plot_importance(feature_importance)
            st.pyplot(fig_importance, width='stretch')
            
            # Feature importance interpretation
            with st.expander("📖 How to Interpret Feature Importance", expanded=False):
                st.markdown("""
                **What this shows**: Which clinical factors the model considers most important for prediction.
                
                **Understanding the Chart**:
                - **Longer bars**: More important features for the model's predictions.
                - **Top features**: These are the key factors the model uses to assess risk.
                
                **Clinical Interpretation**:
                - **High importance features**: These are the primary risk factors the model relies on.
                  Review if these align with known clinical risk factors for Alzheimer's.
                - **Clinical validation**: Check if the top features make clinical sense.
                  For example, MMSE scores, age, and family history are expected to be important.
                - **Unexpected features**: If unusual features rank high, investigate why - 
                  could indicate data quality issues or interesting clinical relationships.
                
                **Use this to**: Understand which patient factors you should pay most attention to 
                when using the model for risk assessment.
                """)
        
        # Feature importance table (already displayed above in plot)
        st.subheader("🔍 Feature Importance Table")
        st.dataframe(feature_importance.head(15), width='stretch')
        
        # Hyperparameters
        if results.get('best_params'):
            st.subheader("⚙️ Best Hyperparameters")
            params_df = pd.DataFrame({
                'Parameter': list(results['best_params'].keys()),
                'Value': list(results['best_params'].values())
            })
            st.dataframe(params_df, width='stretch')
        
        # Threshold optimization
        st.subheader("🎯 Threshold Optimization")
        from models.utils import find_optimal_threshold
        optimal_thresh, optimal_metrics = find_optimal_threshold(
            y_test, y_proba, metric='f1', min_recall=0.90
        )
        
        st.write(f"**Optimal Threshold (≥90% sensitivity):** {optimal_thresh:.4f}")
        st.write(f"**Precision at optimal threshold:** {optimal_metrics['precision']:.4f}")
        st.write(f"**Sensitivity at optimal threshold:** {optimal_metrics['recall']:.4f}")
        st.write(f"**F1-Score at optimal threshold:** {optimal_metrics['f1']:.4f}")
        
        # Threshold interpretation
        with st.expander("📖 Understanding Threshold Optimization", expanded=False):
            st.markdown("""
            **What this shows**: The optimal probability cutoff for making predictions, 
            optimized to meet clinical goals (≥90% sensitivity).
            
            **Understanding Thresholds**:
            - **Default (0.5)**: Standard cutoff - if probability ≥ 50%, predict Alzheimer's.
            - **Lower threshold (e.g., 0.3)**: More sensitive - flags more patients as high risk.
              *Clinical impact*: Catches more true cases but also more false positives.
            - **Higher threshold (e.g., 0.7)**: More specific - only flags high-probability cases.
              *Clinical impact*: Fewer false alarms but may miss some true cases.
            
            **Clinical Goal**: We optimize for ≥90% sensitivity to minimize missed cases.
            This means we're willing to accept more false positives (which can be ruled out 
            with further testing) to ensure we don't miss patients who need early intervention.
            
            **How to Use**: When making predictions, use this threshold. Patients with probability 
            above this threshold should be flagged for further evaluation.
            """)
        
    else:
        st.info("👈 Select a model and click 'Train Model' to get started!")
        st.markdown("""
        ### Available Models:
        
        1. **Logistic Regression**: Highly interpretable linear model with coefficient-based feature importance
        2. **Decision Tree**: Interpretable tree-based model with clear decision rules
        3. **Random Forest**: Ensemble of trees with robust performance
        4. **Gradient Boosting**: Advanced ensemble method with strong predictive power
        
        ### Features:
        - Comprehensive performance metrics
        - Visualizations (Confusion Matrix, ROC, PR curves)
        - Feature importance analysis
        - Hyperparameter tuning (optional)
        - Threshold optimization for clinical recall targets
        """)

