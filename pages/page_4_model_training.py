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
    st.title("🤖 Model Training")
    st.markdown("---")
    
    # Sidebar for model selection and settings
    st.sidebar.header("⚙️ Model Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
        index=0
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
        
        # Metrics display
        metrics = results['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['Precision']:.4f}")
        with col3:
            st.metric("Recall (Sensitivity)", f"{metrics['Recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
        
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("ROC-AUC", f"{metrics.get('ROC-AUC', 0):.4f}")
        with col6:
            st.metric("PR-AUC", f"{metrics.get('PR-AUC', 0):.4f}")
        with col7:
            st.metric("False Negatives", metrics['False Negatives'], 
                     delta=f"-{metrics['False Negatives']}" if metrics['False Negatives'] > 0 else None)
        with col8:
            st.metric("False Positives", metrics['False Positives'])
        
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
        
        with col2:
            fig_roc = plot_roc(y_test, y_proba, f"{model_type}")
            st.pyplot(fig_roc, width='stretch')
        
        # Get feature importance before using it
        feature_importance = results['feature_importance']
        
        # Plot PR curve and feature importance side by side
        col3, col4 = st.columns(2)
        with col3:
            fig_pr = plot_pr(y_test, y_proba, f"{model_type}")
            st.pyplot(fig_pr, width='stretch')
        
        with col4:
            fig_importance = plot_importance(feature_importance)
            st.pyplot(fig_importance, width='stretch')
        
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
        
        st.write(f"**Optimal Threshold (≥90% recall):** {optimal_thresh:.4f}")
        st.write(f"**Precision at optimal threshold:** {optimal_metrics['precision']:.4f}")
        st.write(f"**Recall at optimal threshold:** {optimal_metrics['recall']:.4f}")
        st.write(f"**F1-Score at optimal threshold:** {optimal_metrics['f1']:.4f}")
        
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

