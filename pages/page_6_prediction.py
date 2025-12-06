"""
Page 6: Prediction
Make predictions for individual patients
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def show():
    st.title("🎯 Patient Prediction")
    st.markdown("---")
    
    # Check if model is trained
    if st.session_state.trained_model is None:
        st.warning("⚠️ No model trained yet. Please go to **Model Training** page first to train a model.")
        st.info("""
        Once you've trained a model, you can use this page to:
        - Input patient features
        - Get instant predictions
        - See probability scores
        - Understand feature contributions
        """)
        st.stop()
    
    model = st.session_state.trained_model
    data_dict = st.session_state.data_dict
    feature_names = data_dict['feature_names']
    model_type = st.session_state.get('model_type', 'Unknown')
    
    # Load default dataset to get feature ranges
    try:
        default_data = pd.read_csv("alzheimers_disease_data.csv")
        default_data = default_data.drop(['Diagnosis', 'PatientID', 'DoctorInCharge'], axis=1, errors='ignore')
    except:
        default_data = None
    
    st.header("📝 Patient Information")
    st.markdown("Enter patient features below. You can use default values or customize them.")
    
    # Create input form
    patient_data = {}
    
    # Organize features into categories
    numerical_features = []
    categorical_features = []
    
    for feat in feature_names:
        if default_data is not None and feat in default_data.columns:
            if default_data[feat].dtype in ['int64', 'float64']:
                numerical_features.append(feat)
            else:
                categorical_features.append(feat)
        else:
            numerical_features.append(feat)
    
    # Create columns for better layout
    num_cols = 3
    num_rows = (len(numerical_features) + num_cols - 1) // num_cols
    
    st.subheader("Numerical Features")
    for i in range(num_rows):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < len(numerical_features):
                feat = numerical_features[idx]
                with cols[j]:
                    if default_data is not None and feat in default_data.columns:
                        min_val = float(default_data[feat].min())
                        max_val = float(default_data[feat].max())
                        default_val = float(default_data[feat].median())
                        patient_data[feat] = st.number_input(
                            feat,
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            key=f"num_{feat}"
                        )
                    else:
                        patient_data[feat] = st.number_input(
                            feat,
                            value=0.0,
                            key=f"num_{feat}"
                        )
    
    if categorical_features:
        st.subheader("Categorical Features")
        cat_cols = st.columns(min(len(categorical_features), 3))
        for idx, feat in enumerate(categorical_features):
            with cat_cols[idx % 3]:
                if default_data is not None and feat in default_data.columns:
                    unique_vals = sorted(default_data[feat].unique())
                    default_val = int(default_data[feat].median()) if len(unique_vals) > 0 else unique_vals[0]
                    patient_data[feat] = st.selectbox(
                        feat,
                        options=unique_vals,
                        index=unique_vals.index(default_val) if default_val in unique_vals else 0,
                        key=f"cat_{feat}"
                    )
                else:
                    patient_data[feat] = st.number_input(
                        feat,
                        value=0,
                        key=f"cat_{feat}"
                    )
    
    # Prediction button
    if st.button("🔮 Make Prediction", type="primary"):
        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_data])
        
        # Ensure correct order and all features present
        patient_df = patient_df.reindex(columns=feature_names, fill_value=0)
        
        # Scale if needed (for Logistic Regression)
        if model_type == "Logistic Regression" and 'scaler' in data_dict:
            scaler = data_dict['scaler']
            patient_scaled = scaler.transform(patient_df)
            patient_df = pd.DataFrame(patient_scaled, columns=feature_names)
        
        # Make prediction
        try:
            prediction = model.predict(patient_df)[0]
            probability = model.predict_proba(patient_df)[0]
            
            st.header("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Prediction",
                    "Alzheimer" if prediction == 1 else "No Alzheimer",
                    delta=None
                )
            with col2:
                prob_alz = probability[1] if len(probability) > 1 else probability[0]
                st.metric(
                    "Probability (Alzheimer)",
                    f"{prob_alz:.4f}",
                    delta=f"{(prob_alz - 0.5)*100:.1f}%" if prob_alz > 0.5 else None
                )
            with col3:
                prob_no_alz = probability[0] if len(probability) > 1 else 1 - prob_alz
                st.metric(
                    "Probability (No Alzheimer)",
                    f"{prob_no_alz:.4f}",
                    delta=None
                )
            
            # Risk level
            if prob_alz >= 0.7:
                st.error("⚠️ **High Risk**: Strong indication of Alzheimer's disease")
            elif prob_alz >= 0.5:
                st.warning("⚠️ **Moderate Risk**: Some indication of Alzheimer's disease")
            else:
                st.success("✅ **Low Risk**: Low probability of Alzheimer's disease")
            
            # Feature importance for this prediction
            st.subheader("🔍 Feature Contributions")
            
            # Get feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                importances = np.ones(len(feature_names))
            
            # For tree models, we can't easily get per-instance contributions
            # So we show general feature importance
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances,
                'Patient Value': patient_df.iloc[0].values
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(importance_df.head(15), width='stretch')
            
            # Visualizations side by side
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Feature Importance")
                import matplotlib.pyplot as plt
                fig1, ax1 = plt.subplots(figsize=(6, 5))
                top_features = importance_df.head(10)
                ax1.barh(range(len(top_features)), top_features['Importance'])
                ax1.set_yticks(range(len(top_features)))
                ax1.set_yticklabels(top_features['Feature'], fontsize=9)
                ax1.set_xlabel('Importance', fontsize=10)
                ax1.set_title('Top 10 Most Important Features', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig1, width='stretch')
            
            with col2:
                st.markdown("#### Patient Values vs Average")
                # Compare patient values with dataset average
                if default_data is not None:
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    top_features = importance_df.head(10)
                    patient_vals = top_features['Patient Value'].values
                    avg_vals = [default_data[feat].mean() if feat in default_data.columns else 0 
                               for feat in top_features['Feature']]
                    
                    x = np.arange(len(top_features))
                    width = 0.35
                    ax2.barh(x - width/2, patient_vals, width, label='Patient Value', alpha=0.7)
                    ax2.barh(x + width/2, avg_vals, width, label='Dataset Average', alpha=0.7)
                    ax2.set_yticks(x)
                    ax2.set_yticklabels(top_features['Feature'], fontsize=9)
                    ax2.set_xlabel('Value', fontsize=10)
                    ax2.set_title('Patient vs Average Values', fontsize=11)
                    ax2.legend(fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig2, width='stretch')
                else:
                    st.info("Dataset not available for comparison")
            
            # Patient values for top features
            st.subheader("📋 Patient Values (Top Features)")
            top_features = importance_df.head(10)
            top_patient_values = pd.DataFrame({
                'Feature': top_features['Feature'],
                'Value': top_features['Patient Value']
            })
            st.dataframe(top_patient_values, width='stretch')
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)
    
    else:
        st.info("👆 Fill in the patient information above and click 'Make Prediction' to get results.")
        
        # Show example/default values
        if default_data is not None:
            with st.expander("📋 View Default/Example Values"):
                example_row = default_data.iloc[0]
                st.dataframe(example_row.to_frame('Example Value'), width='stretch')

