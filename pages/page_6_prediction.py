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
    st.title("🎯 Patient Risk Assessment")
    st.markdown("---")
    
    # Return to home button
    if st.button("🏠 Return to Home", type="secondary"):
        st.session_state.current_page = "📖 Introduction"
        st.rerun()
    
    st.markdown("---")
    
    # Clinical context
    st.info("""
    **Clinical Context**: This page allows you to assess individual patient risk for Alzheimer's disease. 
    Enter the patient's clinical information to get a risk assessment with probability scores and 
    explanations of which factors contribute most to the assessment.
    
    **Important**: This tool supports clinical decision-making but does not replace comprehensive 
    clinical assessment. Always interpret results in the context of the full patient evaluation.
    """)
    
    # Check if model is trained
    if st.session_state.trained_model is None:
        st.warning("⚠️ No model trained yet. Please go to **Model Training** page first to train a model.")
        st.info("""
        Once you've trained a model, you can use this page to:
        - Enter individual patient clinical information
        - Get risk assessments with probability scores
        - See which factors contribute most to the risk assessment
        - Understand the clinical reasoning behind the prediction
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
    
    st.header("📝 Patient Clinical Information")
    st.markdown("""
    Enter the patient's clinical information below. Default values are provided based on typical 
    patient data, but you should customize them with the actual patient's values for accurate assessment.
    
    **Note**: More accurate input data leads to more reliable risk assessments.
    """)
    
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
            
            # Risk level with clinical interpretation
            if prob_alz >= 0.7:
                st.error("⚠️ **High Risk**: Strong indication of Alzheimer's disease")
                st.markdown("""
                **Clinical Recommendation**: 
                - Consider comprehensive neuropsychological evaluation
                - Review cognitive assessments and functional status
                - Discuss with patient and family about further diagnostic workup
                - Consider referral to neurology or memory clinic
                """)
            elif prob_alz >= 0.5:
                st.warning("⚠️ **Moderate Risk**: Some indication of Alzheimer's disease")
                st.markdown("""
                **Clinical Recommendation**: 
                - Monitor closely with follow-up assessments
                - Consider additional cognitive screening
                - Review risk factors and consider preventive interventions
                - Discuss concerns with patient and family
                """)
            else:
                st.success("✅ **Low Risk**: Low probability of Alzheimer's disease")
                st.markdown("""
                **Clinical Recommendation**: 
                - Continue routine monitoring
                - Address any modifiable risk factors
                - Maintain regular follow-up as clinically indicated
                """)
            
            # Interpretation guide
            with st.expander("📖 Understanding Risk Assessment", expanded=False):
                st.markdown("""
                **What the Probability Means**:
                - **Probability Score**: The model's estimate of how likely this patient has Alzheimer's disease, 
                  based on the clinical factors entered.
                - **0.0-0.3 (0-30%)**: Low risk - unlikely to have Alzheimer's disease.
                - **0.3-0.5 (30-50%)**: Low-moderate risk - some concerning factors present.
                - **0.5-0.7 (50-70%)**: Moderate-high risk - concerning factors present, further evaluation recommended.
                - **0.7-1.0 (70-100%)**: High risk - strong indication, comprehensive evaluation recommended.
                
                **Important Considerations**:
                - **Not a diagnosis**: This is a risk assessment tool, not a diagnostic test.
                - **Clinical context matters**: Interpret results in the context of the full patient evaluation.
                - **False positives possible**: Some patients may be flagged who don't have the disease 
                  (this is intentional to minimize missed cases).
                - **False negatives possible**: Some patients with the disease may not be flagged 
                  (though we optimize to minimize this).
                - **Use with clinical judgment**: Always combine with comprehensive clinical assessment.
                
                **Next Steps**: Use the feature contributions below to understand which specific 
                factors are driving this patient's risk assessment.
                """)
            
            # Feature importance for this prediction
            st.subheader("🔍 Factors Contributing to Risk Assessment")
            st.markdown("""
            The following shows which clinical factors are most important for this patient's risk assessment. 
            This helps you understand which specific patient characteristics are driving the prediction.
            """)
            
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
                st.markdown("#### Most Important Clinical Factors")
                
                # Interpretation guide
                with st.expander("📖 How to Interpret This Chart", expanded=False):
                    st.markdown("""
                    **What this shows**: Which clinical factors the model considers most important 
                    for this patient's risk assessment.
                    
                    **Understanding the Chart**:
                    - **Y-axis**: Clinical factors, ordered by importance.
                    - **X-axis**: Importance score - how much the model relies on this factor.
                    - **Longer bars**: More important factors for this patient.
                    
                    **Clinical Use**: 
                    - Focus your clinical assessment on the top-ranked factors.
                    - Validate that these factors align with your clinical assessment.
                    - Use this to explain the risk assessment to patients and families.
                    """)
                
                import matplotlib.pyplot as plt
                fig1, ax1 = plt.subplots(figsize=(6, 5))
                top_features = importance_df.head(10)
                ax1.barh(range(len(top_features)), top_features['Importance'])
                ax1.set_yticks(range(len(top_features)))
                ax1.set_yticklabels(top_features['Feature'], fontsize=9)
                ax1.set_xlabel('Importance', fontsize=10)
                ax1.set_title('Top 10 Most Important Clinical Factors', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig1, width='stretch')
            
            with col2:
                st.markdown("#### Patient Values vs Population Average")
                
                # Interpretation guide
                with st.expander("📖 How to Interpret This Chart", expanded=False):
                    st.markdown("""
                    **What this shows**: How this patient's values compare to the average patient 
                    in the dataset for the most important clinical factors.
                    
                    **Understanding the Chart**:
                    - **Blue bars**: This patient's actual values.
                    - **Orange bars**: Average values from the dataset.
                    - **Differences**: Show how this patient differs from the average.
                    
                    **Clinical Interpretation**:
                    - **Patient values higher than average**: May indicate increased risk 
                      (depending on the factor - e.g., age, blood pressure).
                    - **Patient values lower than average**: May indicate decreased risk 
                      (e.g., higher MMSE scores, better cognitive function).
                    - **Use this to**: Identify which of this patient's specific values 
                      are most concerning compared to typical patients.
                    """)
                
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
            st.subheader("📋 Patient Clinical Values (Most Important Factors)")
            st.markdown("""
            Review the patient's actual values for the factors that most influence their risk assessment.
            """)
            top_features = importance_df.head(10)
            top_patient_values = pd.DataFrame({
                'Clinical Factor': top_features['Feature'],
                'Patient Value': top_features['Patient Value']
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

