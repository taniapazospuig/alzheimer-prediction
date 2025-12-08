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
    Enter the patient's clinical information to get a risk assessment with probability scores.
    
    **Important**: This tool supports clinical decision-making but does not replace comprehensive 
    clinical assessment. Always interpret results in the context of the full patient evaluation.
    
    **Want to understand which factors contributed to the prediction?** Visit the **Explainability** page 
    after making a prediction here to see detailed explanations using SHAP or LIME methods.
    """)
    
    # Check if model is trained
    if st.session_state.trained_model is None:
        st.warning("⚠️ No model trained yet. Please go to **Model Training** page first to train a model.")
        st.info("""
        Once you've trained a model, you can use this page to:
        - Enter individual patient clinical information
        - Get risk assessments with probability scores
        - Understand the clinical interpretation of the results
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
    Enter the patient's clinical information below using the sliders and dropdowns. Each feature includes 
    a description to guide you. Default values are set to the median from the dataset, but you should 
    customize them with the actual patient's values for accurate assessment.
    
    **Note**: More accurate input data leads to more reliable risk assessments.
    """)
    
    # Feature descriptions based on official documentation
    feature_descriptions = {
        # Demographic Details
        'Age': 'Age of the patient (60-90 years)',
        'Gender': 'Gender: 0 = Male, 1 = Female',
        'Ethnicity': 'Ethnicity: 0 = Caucasian, 1 = African American, 2 = Asian, 3 = Other',
        'EducationLevel': 'Education Level: 0 = None, 1 = High School, 2 = Bachelor\'s, 3 = Higher',
        
        # Lifestyle Factors
        'BMI': 'Body Mass Index (15-40)',
        'Smoking': 'Smoking status: 0 = No, 1 = Yes',
        'AlcoholConsumption': 'Weekly alcohol consumption in units (0-20)',
        'PhysicalActivity': 'Weekly physical activity in hours (0-10)',
        'DietQuality': 'Diet quality score (0-10)',
        'SleepQuality': 'Sleep quality score (4-10)',
        
        # Medical History
        'FamilyHistoryAlzheimers': 'Family history of Alzheimer\'s Disease: 0 = No, 1 = Yes',
        'CardiovascularDisease': 'Presence of cardiovascular disease: 0 = No, 1 = Yes',
        'Diabetes': 'Presence of diabetes: 0 = No, 1 = Yes',
        'Depression': 'Presence of depression: 0 = No, 1 = Yes',
        'HeadInjury': 'History of head injury: 0 = No, 1 = Yes',
        'Hypertension': 'Presence of hypertension: 0 = No, 1 = Yes',
        
        # Clinical Measurements
        'SystolicBP': 'Systolic blood pressure in mmHg (90-180)',
        'DiastolicBP': 'Diastolic blood pressure in mmHg (60-120)',
        'CholesterolTotal': 'Total cholesterol levels in mg/dL (150-300)',
        'CholesterolLDL': 'Low-density lipoprotein cholesterol in mg/dL (50-200)',
        'CholesterolHDL': 'High-density lipoprotein cholesterol in mg/dL (20-100)',
        'CholesterolTriglycerides': 'Triglycerides levels in mg/dL (50-400)',
        
        # Cognitive and Functional Assessments
        'MMSE': 'Mini-Mental State Examination score (0-30). Lower scores indicate cognitive impairment.',
        'FunctionalAssessment': 'Functional assessment score (0-10). Lower scores indicate greater impairment.',
        'MemoryComplaints': 'Presence of memory complaints: 0 = No, 1 = Yes',
        'BehavioralProblems': 'Presence of behavioral problems: 0 = No, 1 = Yes',
        'ADL': 'Activities of Daily Living score (0-10). Lower scores indicate greater impairment.',
        
        # Symptoms
        'Confusion': 'Presence of confusion: 0 = No, 1 = Yes',
        'Disorientation': 'Presence of disorientation: 0 = No, 1 = Yes',
        'PersonalityChanges': 'Presence of personality changes: 0 = No, 1 = Yes',
        'DifficultyCompletingTasks': 'Presence of difficulty completing tasks: 0 = No, 1 = Yes',
        'Forgetfulness': 'Presence of forgetfulness: 0 = No, 1 = Yes',
    }
    
    # Feature ranges from documentation
    feature_ranges = {
        'Age': (60, 90),
        'BMI': (15, 40),
        'AlcoholConsumption': (0, 20),
        'PhysicalActivity': (0, 10),
        'DietQuality': (0, 10),
        'SleepQuality': (4, 10),
        'SystolicBP': (90, 180),
        'DiastolicBP': (60, 120),
        'CholesterolTotal': (150, 300),
        'CholesterolLDL': (50, 200),
        'CholesterolHDL': (20, 100),
        'CholesterolTriglycerides': (50, 400),
        'MMSE': (0, 30),
        'FunctionalAssessment': (0, 10),
        'ADL': (0, 10),
    }
    
    # Categorical feature options
    categorical_options = {
        'Gender': {0: 'Male', 1: 'Female'},
        'Ethnicity': {0: 'Caucasian', 1: 'African American', 2: 'Asian', 3: 'Other'},
        'EducationLevel': {0: 'None', 1: 'High School', 2: 'Bachelor\'s', 3: 'Higher'},
        'Smoking': {0: 'No', 1: 'Yes'},
        'FamilyHistoryAlzheimers': {0: 'No', 1: 'Yes'},
        'CardiovascularDisease': {0: 'No', 1: 'Yes'},
        'Diabetes': {0: 'No', 1: 'Yes'},
        'Depression': {0: 'No', 1: 'Yes'},
        'HeadInjury': {0: 'No', 1: 'Yes'},
        'Hypertension': {0: 'No', 1: 'Yes'},
        'MemoryComplaints': {0: 'No', 1: 'Yes'},
        'BehavioralProblems': {0: 'No', 1: 'Yes'},
        'Confusion': {0: 'No', 1: 'Yes'},
        'Disorientation': {0: 'No', 1: 'Yes'},
        'PersonalityChanges': {0: 'No', 1: 'Yes'},
        'DifficultyCompletingTasks': {0: 'No', 1: 'Yes'},
        'Forgetfulness': {0: 'No', 1: 'Yes'},
    }
    
    # Organize features into categories
    categories = {
        'Demographic Details': ['Age', 'Gender', 'Ethnicity', 'EducationLevel'],
        'Lifestyle Factors': ['BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality'],
        'Medical History': ['FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension'],
        'Clinical Measurements': ['SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides'],
        'Cognitive and Functional Assessments': ['MMSE', 'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL'],
        'Symptoms': ['Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness'],
    }
    
    # Create input form
    patient_data = {}
    
    # Process each category
    for category_name, features_in_category in categories.items():
        # Filter to only include features that are actually in feature_names
        features_in_category = [f for f in features_in_category if f in feature_names]
        
        if not features_in_category:
            continue
            
        st.subheader(f"📋 {category_name}")
        
        # Create columns (2 columns for better layout)
        num_cols = 2
        num_rows = (len(features_in_category) + num_cols - 1) // num_cols
        
        for i in range(num_rows):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                idx = i * num_cols + j
                if idx < len(features_in_category):
                    feat = features_in_category[idx]
                    with cols[j]:
                        # Get description
                        description = feature_descriptions.get(feat, f'{feat} value')
                        
                        # Determine if categorical or continuous
                        if feat in categorical_options:
                            # Categorical feature - use selectbox
                            options_dict = categorical_options[feat]
                            options_list = list(options_dict.keys())
                            labels_list = [options_dict[k] for k in options_list]
                            
                            # Get default value
                            if default_data is not None and feat in default_data.columns:
                                default_val = int(default_data[feat].median())
                                if default_val not in options_list:
                                    default_val = options_list[0]
                                default_idx = options_list.index(default_val)
                            else:
                                default_idx = 0
                            
                            selected_label = st.selectbox(
                                feat,
                                options=labels_list,
                                index=default_idx,
                                help=description,
                                key=f"pred_{feat}"
                            )
                            # Get the numeric value corresponding to the selected label
                            patient_data[feat] = options_list[labels_list.index(selected_label)]
                        else:
                            # Continuous feature - use slider
                            if feat in feature_ranges:
                                min_val, max_val = feature_ranges[feat]
                                # Convert to float to ensure type consistency
                                min_val = float(min_val)
                                max_val = float(max_val)
                            elif default_data is not None and feat in default_data.columns:
                                min_val = float(default_data[feat].min())
                                max_val = float(default_data[feat].max())
                            else:
                                min_val, max_val = 0.0, 100.0
                            
                            # Get default value
                            if default_data is not None and feat in default_data.columns:
                                default_val = float(default_data[feat].median())
                            else:
                                default_val = float((min_val + max_val) / 2)
                            
                            # Determine step size (ensure it's float)
                            range_size = max_val - min_val
                            if range_size < 10:
                                step = 0.1
                            elif range_size < 100:
                                step = 1.0
                            else:
                                step = 5.0
                            
                            # Ensure all values are float for consistency
                            patient_data[feat] = st.slider(
                                feat,
                                min_value=min_val,
                                max_value=max_val,
                                value=default_val,
                                step=step,
                                help=description,
                                key=f"pred_{feat}"
                            )
    
    # Handle any features not in categories
    uncategorized_features = [f for f in feature_names if f not in [item for sublist in categories.values() for item in sublist]]
    if uncategorized_features:
        st.subheader("📋 Other Features")
        for feat in uncategorized_features:
            if default_data is not None and feat in default_data.columns:
                description = feature_descriptions.get(feat, f'{feat} value')
                
                if feat in feature_ranges:
                    min_val, max_val = feature_ranges[feat]
                    min_val = float(min_val)
                    max_val = float(max_val)
                else:
                    min_val = float(default_data[feat].min())
                    max_val = float(default_data[feat].max())
                
                default_val = float(default_data[feat].median())
                
                range_size = max_val - min_val
                step = 0.1 if range_size < 10 else (1.0 if range_size < 100 else 5.0)
                
                patient_data[feat] = st.slider(
                    feat,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=step,
                    help=description,
                    key=f"pred_{feat}"
                )
            else:
                patient_data[feat] = 0.0
    
    # Prediction button
    if st.button("🔮 Make Prediction", type="primary"):
        # Store patient data in session state for use in explainability page
        st.session_state.patient_data_for_explainability = patient_data.copy()
        
        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_data])
        
        # Ensure correct order and all features present
        patient_df = patient_df.reindex(columns=feature_names, fill_value=0)
        
        # Store unscaled patient_df for explainability (before scaling)
        st.session_state.patient_df_for_explainability = patient_df.copy()
        
        # Scale if needed (for Logistic Regression)
        if model_type == "Logistic Regression" and 'scaler' in data_dict:
            scaler = data_dict['scaler']
            patient_scaled = scaler.transform(patient_df)
            patient_df = pd.DataFrame(patient_scaled, columns=feature_names)
        
        # Make prediction
        try:
            prediction = model.predict(patient_df)[0]
            probability = model.predict_proba(patient_df)[0]
            
            # Store prediction results for explainability page
            st.session_state.last_prediction = prediction
            st.session_state.last_probability = probability
            
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
                
                **Next Steps**: Visit the **Explainability** page to see detailed explanations of which 
                specific factors contributed to this patient's risk assessment using SHAP or LIME methods.
                """)
            
            st.markdown("---")
            
            # Link to explainability page
            st.info("""
            **🔍 Want to understand which factors contributed to this prediction?** 
            
            Go to the **Explainability** page to see:
            - Which clinical factors are most important for this specific patient
            - How each factor pushed the prediction toward or away from Alzheimer's
            - Detailed visualizations using SHAP or LIME methods
            
            **💡 Tip**: The patient data you just entered has been saved and will be automatically loaded 
            in the Explainability page. You can modify it there if needed, or use it as-is to get explanations 
            for this exact prediction.
            """)
            
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

