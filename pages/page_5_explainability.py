"""
Page 5: Explainability
Understand model predictions using SHAP and LIME
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def show():
    st.title("🔍 Model Explainability")
    st.markdown("---")
    
    # Return to home button
    if st.button("🏠 Return to Home", type="secondary"):
        st.session_state.current_page = "📖 Introduction"
        st.rerun()
    
    st.markdown("---")
    
    # Clinical context
    st.info("""
    **Clinical Context**: This page helps you understand how the model makes predictions. 
    You can see which clinical factors are most important overall (Global) or for specific 
    patients (Local). All visualizations include interpretation guides.
    """)
    
    # Check if model is trained
    if st.session_state.trained_model is None:
        st.warning("⚠️ No model trained yet. Please go to **Model Training** page first to train a model.")
        st.info("""
        Once you've trained a model, you can use this page to:
        - View which clinical factors are most important for predictions
        - Understand why specific patients are flagged as high risk
        - See how individual patient factors contribute to their risk assessment
        """)
        st.stop()
    
    model = st.session_state.trained_model
    model_type = st.session_state.get('model_type', 'Unknown')
    data_dict = st.session_state.data_dict
    
    st.sidebar.header("⚙️ Explainability Settings")
    
    # Method selection
    explainability_method = st.sidebar.selectbox(
        "Select Explainability Method",
        ["SHAP", "LIME"],
        help="Choose the explainability technique to use"
    )
    
    # Analysis type
    analysis_type = st.sidebar.radio(
        "Analysis Type",
        ["Global (Overall Model)", "Local (Individual Predictions)"],
        help="Global: Understand the model overall. Local: Understand specific predictions."
    )
    
    # Main content
    if analysis_type == "Global (Overall Model)":
        st.header("🌐 Global Explainability")
        st.markdown("""
        **What this shows**: Which clinical factors are most important for the model's predictions 
        across all patients. This helps you understand the model's overall decision-making process.
        
        **Clinical Use**: Use this to validate that the model is focusing on clinically relevant 
        risk factors (e.g., MMSE scores, age, family history) rather than spurious correlations.
        """)
        
        if explainability_method == "SHAP":
            show_shap_global(model, data_dict, model_type)
        elif explainability_method == "LIME":
            st.info("LIME is primarily designed for local explanations. Please select 'Local' analysis type for LIME.")
    
    else:  # Local explanations
        st.header("🔬 Local Explainability")
        st.markdown("""
        **What this shows**: Why the model made a specific prediction for an individual patient. 
        Enter patient information below, make a prediction, and then see which factors contributed to that prediction.
        
        **Clinical Use**: Use this to:
        - Explain predictions to patients and families
        - Identify which specific factors are driving a patient's risk
        - Validate that the model's reasoning aligns with clinical assessment
        """)
        
        # Load default dataset to get feature ranges
        try:
            default_data = pd.read_csv("alzheimers_disease_data.csv")
            default_data = default_data.drop(['Diagnosis', 'PatientID', 'DoctorInCharge'], axis=1, errors='ignore')
        except:
            default_data = None
        
        feature_names = data_dict['feature_names']
        
        # Check if patient data is available from prediction page
        stored_patient_data = st.session_state.get('patient_data_for_explainability', None)
        stored_patient_df = st.session_state.get('patient_df_for_explainability', None)
        
        st.subheader("📝 Enter Patient Clinical Information")
        
        # Show info if stored data is available
        if stored_patient_data is not None:
            st.success("✅ **Patient data from Prediction page is loaded!** You can modify the values below or use them as-is.")
            if st.button("🔄 Clear stored data and start fresh", type="secondary"):
                st.session_state.patient_data_for_explainability = None
                st.session_state.patient_df_for_explainability = None
                st.session_state.last_prediction = None
                st.session_state.last_probability = None
                st.rerun()
            st.markdown("---")
        else:
            st.markdown("""
            Enter the patient's clinical information below using the sliders and dropdowns. Each feature includes 
            a description to guide you. After making a prediction, you'll see which factors contributed to that specific prediction.
            
            **Tip**: If you've made a prediction on the **Prediction** page, that data will be automatically loaded here.
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
                
            st.markdown(f"#### {category_name}")
            
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
                                
                                # Get default value - prefer stored data, then dataset median
                                if stored_patient_data is not None and feat in stored_patient_data:
                                    default_val = int(stored_patient_data[feat])
                                    if default_val not in options_list:
                                        default_val = options_list[0]
                                    default_idx = options_list.index(default_val)
                                elif default_data is not None and feat in default_data.columns:
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
                                    key=f"explain_{feat}"
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
                                
                                # Get default value - prefer stored data, then dataset median
                                if stored_patient_data is not None and feat in stored_patient_data:
                                    default_val = float(stored_patient_data[feat])
                                    # Ensure value is within range
                                    default_val = max(min_val, min(max_val, default_val))
                                elif default_data is not None and feat in default_data.columns:
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
                                    key=f"explain_{feat}"
                                )
        
        # Handle any features not in categories
        uncategorized_features = [f for f in feature_names if f not in [item for sublist in categories.values() for item in sublist]]
        if uncategorized_features:
            st.markdown("#### Other Features")
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
                    
                    # Get default value - prefer stored data, then dataset median
                    if stored_patient_data is not None and feat in stored_patient_data:
                        default_val = float(stored_patient_data[feat])
                        default_val = max(min_val, min(max_val, default_val))
                    else:
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
                        key=f"explain_{feat}"
                    )
                else:
                    # Use stored value if available, otherwise 0.0
                    if stored_patient_data is not None and feat in stored_patient_data:
                        patient_data[feat] = float(stored_patient_data[feat])
                    else:
                        patient_data[feat] = 0.0
        
        # Prediction button
        if st.button("🔮 Make Prediction & Explain", type="primary"):
            # Convert to DataFrame
            patient_df = pd.DataFrame([patient_data])
            
            # Ensure correct order and all features present
            patient_df = patient_df.reindex(columns=feature_names, fill_value=0)
            
            # Scale if needed (for Logistic Regression)
            if model_type == "Logistic Regression" and 'scaler' in data_dict:
                scaler = data_dict['scaler']
                patient_scaled = scaler.transform(patient_df)
                patient_df = pd.DataFrame(patient_scaled, columns=feature_names)
            
            # Make prediction FIRST
            prediction = None
            probability = None
            prediction_success = False
            
            try:
                prediction = model.predict(patient_df)[0]
                probability = model.predict_proba(patient_df)[0]
                prediction_success = True
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.exception(e)
                st.stop()  # Stop execution if prediction fails
            
            # Only show prediction results if prediction succeeded
            if prediction_success:
                st.subheader("📊 Prediction Results")
                
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
                        delta=None
                    )
                with col3:
                    prob_no_alz = probability[0] if len(probability) > 1 else 1 - prob_alz
                    st.metric(
                        "Probability (No Alzheimer)",
                        f"{prob_no_alz:.4f}",
                        delta=None
                    )
                
                st.markdown("---")
                
                # Show explanation ONLY after prediction is successfully shown
                st.subheader("🔍 Explanation of This Prediction")
                try:
                    if explainability_method == "SHAP":
                        show_shap_local(model, data_dict, patient_df, model_type)
                    elif explainability_method == "LIME":
                        show_lime_local(model, data_dict, patient_df, model_type)
                except Exception as e:
                    st.error(f"❌ Error generating {explainability_method} explanations: {str(e)}")
                    st.exception(e)
        
        else:
            st.info("👆 Fill in the patient information above and click 'Make Prediction & Explain' to see the explanation.")


def show_shap_global(model, data_dict, model_type):
    """Show SHAP global explanations."""
    try:
        import shap
        
        st.subheader("SHAP Global Explanations")
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** provides a mathematically rigorous way to measure 
        how much each clinical factor contributes to predictions. It's based on game theory principles.
        
        **Clinical Interpretation**: SHAP values show which factors the model considers most important 
        when making predictions. Higher absolute SHAP values indicate more important factors.
        """)
        
        X_test = data_dict['X_test']
        sample_size = min(100, len(X_test))
        X_test_sample = X_test.iloc[:sample_size]
        
        with st.spinner("Calculating SHAP values... This may take a moment."):
            # Create explainer
            if model_type in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X_test_sample)
            
            shap_values = explainer(X_test_sample)
            
            # For binary classification, use class 1 (Alzheimer)
            if len(shap_values.shape) == 3:
                shap_values_class1 = shap_values.values[:, :, 1]
            else:
                shap_values_class1 = shap_values.values
            
            st.success("✅ SHAP values calculated!")
            
            # Summary plot and Bar plot side by side
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Summary Plot")
                st.markdown("Shows feature importance and impact direction")
                
                # Interpretation guide
                with st.expander("📖 How to Interpret This Plot", expanded=False):
                    st.markdown("""
                    **What this shows**: Which clinical factors are most important and how they affect predictions.
                    
                    **Understanding the Plot**:
                    - **Y-axis**: Clinical factors, ordered by importance (most important at top).
                    - **X-axis**: SHAP value - how much this factor pushes the prediction.
                    - **Color**: The actual value of the factor (red = high, blue = low).
                    
                    **Clinical Interpretation**:
                    - **Points to the right (positive SHAP)**: This factor increases Alzheimer's risk.
                    - **Points to the left (negative SHAP)**: This factor decreases Alzheimer's risk.
                    - **Spread of points**: Shows how the factor's impact varies across patients.
                    - **Red points on right**: High values of this factor increase risk.
                    - **Blue points on right**: Low values of this factor increase risk.
                    
                    **Example**: If MMSE scores are at the top with points mostly on the left (blue/red), 
                    it means lower MMSE scores (blue) push predictions toward Alzheimer's, 
                    which aligns with clinical knowledge.
                    """)
                plt.ioff()  # Turn off interactive mode
                try:
                    shap.summary_plot(shap_values_class1, X_test_sample, show=False)
                    fig1 = plt.gcf()  # Get current figure
                    st.pyplot(fig1, width='stretch')
                    plt.close(fig1)  # Close to free memory
                except Exception as e:
                    # Fallback to manual plot - select features with high variability
                    fig1, ax = plt.subplots(figsize=(6, 5))
                    mean_shap = np.abs(shap_values_class1).mean(0)
                    shap_std = np.std(shap_values_class1, axis=0)
                    # Combine importance and variability
                    mean_shap_norm = (mean_shap - mean_shap.min()) / (mean_shap.max() - mean_shap.min() + 1e-10)
                    shap_std_norm = (shap_std - shap_std.min()) / (shap_std.max() - shap_std.min() + 1e-10)
                    combined_score = 0.6 * mean_shap_norm + 0.4 * shap_std_norm
                    top_indices = np.argsort(combined_score)[-10:][::-1]
                    ax.barh(range(len(top_indices)), mean_shap[top_indices])
                    ax.set_yticks(range(len(top_indices)))
                    ax.set_yticklabels([X_test_sample.columns[i] for i in top_indices], fontsize=9)
                    ax.set_xlabel('Mean |SHAP|', fontsize=10)
                    ax.set_title('Top 10 Features (High Variability)', fontsize=11)
                    plt.tight_layout()
                    st.pyplot(fig1, width='stretch')
                    plt.close(fig1)
            
            with col2:
                st.markdown("#### Bar Plot (Mean |SHAP|)")
                st.markdown("Mean absolute SHAP values per feature")
                
                # Interpretation guide
                with st.expander("📖 How to Interpret This Plot", expanded=False):
                    st.markdown("""
                    **What this shows**: The average importance of each clinical factor across all patients.
                    
                    **Understanding the Plot**:
                    - **Y-axis**: Clinical factors, ordered by average importance.
                    - **X-axis**: Mean absolute SHAP value - average impact on predictions.
                    - **Longer bars**: More important factors for the model.
                    
                    **Clinical Interpretation**:
                    - **Top factors**: These are the primary risk factors the model uses.
                      Review if these align with known clinical risk factors.
                    - **Clinical validation**: Check if the ranking makes clinical sense.
                      For example, MMSE scores, age, and cognitive assessments should rank high.
                    - **Unexpected rankings**: If unusual factors rank high, investigate why.
                    
                    **Use this to**: Understand which patient factors you should prioritize 
                    when assessing Alzheimer's risk.
                    """)
                try:
                    shap.plots.bar(shap_values[:,:,1] if len(shap_values.shape) == 3 else shap_values, show=False)
                    fig2 = plt.gcf()  # Get current figure
                    st.pyplot(fig2, width='stretch')
                    plt.close(fig2)
                except:
                    # Fallback to manual bar plot - select features with high variability
                    mean_shap = np.abs(shap_values_class1).mean(0)
                    shap_std = np.std(shap_values_class1, axis=0)
                    # Combine importance and variability
                    mean_shap_norm = (mean_shap - mean_shap.min()) / (mean_shap.max() - mean_shap.min() + 1e-10)
                    shap_std_norm = (shap_std - shap_std.min()) / (shap_std.max() - shap_std.min() + 1e-10)
                    combined_score = 0.6 * mean_shap_norm + 0.4 * shap_std_norm
                    fig2, ax = plt.subplots(figsize=(6, 5))
                    top_indices = np.argsort(combined_score)[-10:][::-1]
                    ax.barh(range(len(top_indices)), mean_shap[top_indices])
                    ax.set_yticks(range(len(top_indices)))
                    ax.set_yticklabels([X_test_sample.columns[i] for i in top_indices], fontsize=9)
                    ax.set_xlabel('Mean |SHAP|', fontsize=10)
                    ax.set_title('Top 10 Features (High Variability)', fontsize=11)
                    plt.tight_layout()
                    st.pyplot(fig2, width='stretch')
                    plt.close(fig2)
            
            # Dependence plots for top features with high variability
            st.markdown("#### Scatter Plots")
            st.markdown("Shows how SHAP values depend on feature values (selecting features with high variability)")
            mean_shap = np.abs(shap_values_class1).mean(0)
            # Calculate variability (standard deviation) of SHAP values for each feature
            shap_std = np.std(shap_values_class1, axis=0)
            # Combine importance and variability (weighted score)
            # Use a combination: 0.6 * mean_shap + 0.4 * std_shap (normalized)
            mean_shap_norm = (mean_shap - mean_shap.min()) / (mean_shap.max() - mean_shap.min() + 1e-10)
            shap_std_norm = (shap_std - shap_std.min()) / (shap_std.max() - shap_std.min() + 1e-10)
            combined_score = 0.6 * mean_shap_norm + 0.4 * shap_std_norm
            # Select top 2 features with highest combined score (importance + variability)
            top_2_features = np.argsort(combined_score)[-2:][::-1]
            top_feature_names = [X_test_sample.columns[i] for i in top_2_features]
            
            if len(top_feature_names) >= 2:
                col3, col4 = st.columns(2)
                with col3:
                    try:
                        shap.plots.scatter(shap_values[:, top_2_features[0], 1] if len(shap_values.shape) == 3 else shap_values[:, top_2_features[0]], show=False)
                        fig3 = plt.gcf()  # Get current figure
                        st.pyplot(fig3, width='stretch')
                        plt.close(fig3)
                    except:
                        # Fallback
                        fig3, ax = plt.subplots(figsize=(6, 5))
                        ax.scatter(X_test_sample.iloc[:, top_2_features[0]], shap_values_class1[:, top_2_features[0]], alpha=0.5)
                        ax.set_xlabel(top_feature_names[0], fontsize=10)
                        ax.set_ylabel('SHAP Value', fontsize=10)
                        ax.set_title(f'SHAP Dependence: {top_feature_names[0]}', fontsize=11)
                        plt.tight_layout()
                        st.pyplot(fig3, width='stretch')
                        plt.close(fig3)
                
                with col4:
                    try:
                        shap.plots.scatter(shap_values[:, top_2_features[1], 1] if len(shap_values.shape) == 3 else shap_values[:, top_2_features[1]], show=False)
                        fig4 = plt.gcf()  # Get current figure
                        st.pyplot(fig4, width='stretch')
                        plt.close(fig4)
                    except:
                        # Fallback
                        fig4, ax = plt.subplots(figsize=(6, 5))
                        ax.scatter(X_test_sample.iloc[:, top_2_features[1]], shap_values_class1[:, top_2_features[1]], alpha=0.5)
                        ax.set_xlabel(top_feature_names[1], fontsize=10)
                        ax.set_ylabel('SHAP Value', fontsize=10)
                        ax.set_title(f'SHAP Dependence: {top_feature_names[1]}', fontsize=11)
                        plt.tight_layout()
                        st.pyplot(fig4, width='stretch')
                        plt.close(fig4)
            
            # Mean Absolute SHAP Values table
            st.markdown("#### Mean Absolute SHAP Values Table")
            mean_shap = np.abs(shap_values_class1).mean(0)
            shap_df = pd.DataFrame({
                'Feature': X_test_sample.columns,
                'Mean |SHAP|': mean_shap
            }).sort_values('Mean |SHAP|', ascending=False)
            
            st.dataframe(shap_df.head(15), width='stretch')
            
    except ImportError:
        st.error("SHAP library not installed. Please install it using: `pip install shap`")
    except Exception as e:
        st.error(f"Error generating SHAP explanations: {str(e)}")


def show_shap_local(model, data_dict, patient_df, model_type):
    """Show SHAP local explanations."""
    try:
        import shap
        
        st.markdown("""
        **What this shows**: How each of this patient's specific clinical factors contributed 
        to their individual risk prediction.
        """)
        
        X_sample = patient_df
        
        with st.spinner("Calculating SHAP values..."):
            if model_type in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
                explainer = shap.TreeExplainer(model)
            else:
                # For non-tree models, use a sample from training data as background
                X_train = data_dict['X_train']
                background_sample = X_train.sample(min(100, len(X_train)))
                explainer = shap.Explainer(model, background_sample)
            
            shap_values = explainer(X_sample)
            
            if len(shap_values.shape) == 3:
                shap_values_class1 = shap_values.values[0, :, 1]
            else:
                shap_values_class1 = shap_values.values[0]
            
            # Waterfall and Bar plots side by side
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Waterfall Plot")
                st.markdown("Shows how each feature pushes the prediction")
                
                # Interpretation guide
                with st.expander("📖 How to Interpret This Plot", expanded=False):
                    st.markdown("""
                    **What this shows**: Step-by-step how each clinical factor moves this patient's 
                    risk prediction from the baseline (average risk) to the final prediction.
                    
                    **Understanding the Plot**:
                    - **Starting point (bottom)**: Baseline risk (average patient).
                    - **Each bar**: How one factor pushes the prediction up or down.
                    - **Red bars**: Increase risk (push toward Alzheimer's).
                    - **Blue bars**: Decrease risk (push away from Alzheimer's).
                    - **Final value (top)**: The patient's final risk score.
                    
                    **Clinical Interpretation**:
                    - **Large red bars**: These factors significantly increase this patient's risk.
                    - **Large blue bars**: These factors significantly decrease this patient's risk.
                    - **Order matters**: Factors are ordered by impact (most impactful first).
                    
                    **Use this to**: Explain to patients and families which specific factors 
                    are driving their risk assessment.
                    """)
                # Ensure plt is available and set to non-interactive mode
                plt.ioff()  # Turn off interactive mode
                fig1 = None
                try:
                    shap.waterfall_plot(shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0], show=False)
                    fig1 = plt.gcf()  # Get current figure
                    st.pyplot(fig1, width='stretch')
                    plt.close(fig1)
                    fig1 = None
                except Exception as e:
                    # Fallback to manual plot
                    if fig1 is not None:
                        plt.close(fig1)
                    fig1, ax = plt.subplots(figsize=(6, 5))
                    top_indices = np.argsort(np.abs(shap_values_class1))[-10:][::-1]
                    ax.barh(range(len(top_indices)), shap_values_class1[top_indices])
                    ax.set_yticks(range(len(top_indices)))
                    ax.set_yticklabels([X_sample.columns[i] for i in top_indices], fontsize=9)
                    ax.set_xlabel('SHAP Value', fontsize=10)
                    ax.set_title('Top 10 Feature Contributions', fontsize=11)
                    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                    plt.tight_layout()
                    st.pyplot(fig1, width='stretch')
                    plt.close(fig1)
            
            with col2:
                st.markdown("#### Feature Contributions Bar Chart")
                
                # Interpretation guide
                with st.expander("📖 How to Interpret This Chart", expanded=False):
                    st.markdown("""
                    **What this shows**: Which of this patient's specific factors contributed most 
                    to their risk prediction.
                    
                    **Understanding the Chart**:
                    - **Y-axis**: Clinical factors, ordered by contribution magnitude.
                    - **X-axis**: SHAP value - how much this factor contributed.
                    - **Red bars (positive)**: Increase this patient's risk.
                    - **Blue bars (negative)**: Decrease this patient's risk.
                    - **Longer bars**: More impactful factors for this specific patient.
                    
                    **Clinical Interpretation**:
                    - **Top contributing factors**: These are the key factors driving this patient's risk.
                    - **Positive values**: These patient values increase their risk.
                    - **Negative values**: These patient values decrease their risk.
                    
                    **Use this to**: 
                    - Identify which specific patient factors to focus on in clinical assessment.
                    - Explain predictions to patients and families.
                    - Validate that the model's reasoning aligns with clinical judgment.
                    """)
                
                contributions_df = pd.DataFrame({
                    'Feature': X_sample.columns,
                    'SHAP Value': shap_values_class1
                }).sort_values('SHAP Value', key=abs, ascending=False)
                
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                top_contrib = contributions_df.head(10)
                colors = ['red' if x < 0 else 'green' for x in top_contrib['SHAP Value']]
                ax2.barh(range(len(top_contrib)), top_contrib['SHAP Value'], color=colors, alpha=0.7)
                ax2.set_yticks(range(len(top_contrib)))
                ax2.set_yticklabels(top_contrib['Feature'], fontsize=9)
                ax2.set_xlabel('SHAP Value', fontsize=10)
                ax2.set_title('Top 10 Feature Contributions', fontsize=11)
                ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                plt.tight_layout()
                st.pyplot(fig2, width='stretch')
                plt.close(fig2)
            
            # Feature contributions table
            contributions_df = pd.DataFrame({
                'Feature': X_sample.columns,
                'SHAP Value': shap_values_class1
            }).sort_values('SHAP Value', key=abs, ascending=False)
            
            st.markdown("#### Feature Contributions Table")
            st.dataframe(contributions_df.head(15), width='stretch')
            
    except ImportError:
        st.error("SHAP library not installed. Please install it using: `pip install shap`")
    except Exception as e:
        st.error(f"Error generating SHAP local explanations: {str(e)}")


def show_lime_local(model, data_dict, patient_df, model_type):
    """Show LIME local explanations."""
    try:
        import lime
        import lime.lime_tabular
        
        st.markdown("""
        **What this shows**: How each of this patient's specific clinical factors contributed 
        to their individual risk prediction using LIME (Local Interpretable Model-agnostic Explanations).
        """)
        
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_sample = patient_df.iloc[0]
        feature_names = data_dict['feature_names']
        
        with st.spinner("Creating LIME explainer..."):
            try:
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train.values,
                    feature_names=feature_names,
                    class_names=['No Alzheimer', 'Alzheimer'],
                    mode='classification',
                    training_labels=y_train.values
                )
            except Exception as e:
                st.error(f"Error creating LIME explainer: {str(e)}")
                raise
        
        with st.spinner("Generating LIME explanation..."):
            try:
                explanation = explainer.explain_instance(
                    X_sample.values,
                    model.predict_proba,
                    num_features=min(15, len(feature_names)),
                    top_labels=1
                )
            except Exception as e:
                st.error(f"Error generating LIME explanation: {str(e)}")
                st.info("This might occur if the model's predict_proba method is not compatible with LIME.")
                raise
        
        # Get explanation as list - check available labels first
        try:
            # Get available labels from the explanation
            available_labels = list(explanation.available_labels())
            if not available_labels:
                st.error("No labels available in LIME explanation.")
                raise ValueError("No labels available in LIME explanation")
            
            # Determine which label to use
            # Prefer label 1 (Alzheimer) if available, otherwise use label 0 (No Alzheimer), 
            # or fall back to the first available label
            if 1 in available_labels:
                label_to_use = 1
            elif 0 in available_labels:
                label_to_use = 0
            else:
                # Use the first available label (usually the predicted class)
                label_to_use = available_labels[0]
            
            exp_list = explanation.as_list(label=label_to_use)
        except KeyError as e:
            available_labels_str = ', '.join(map(str, available_labels)) if 'available_labels' in locals() else 'unknown'
            st.error(f"Error extracting LIME explanation: Label {e} not found. Available labels: [{available_labels_str}]")
            raise
        except Exception as e:
            st.error(f"Error extracting LIME explanation: {str(e)}")
            raise
        
        # Create DataFrame
        exp_df = pd.DataFrame(exp_list, columns=['Feature', 'Contribution'])
        exp_df = exp_df.sort_values('Contribution', key=abs, ascending=False)
        
        # Visualizations side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Feature Contributions Bar Chart")
            try:
                fig1, ax1 = plt.subplots(figsize=(6, 5))
                top_contrib = exp_df.head(10)
                colors = ['red' if v < 0 else 'green' for v in top_contrib['Contribution']]
                ax1.barh(range(len(top_contrib)), top_contrib['Contribution'], color=colors, alpha=0.7)
                ax1.set_yticks(range(len(top_contrib)))
                ax1.set_yticklabels(top_contrib['Feature'], fontsize=9)
                ax1.set_xlabel('Contribution', fontsize=10)
                ax1.set_title('Top 10 Feature Contributions', fontsize=11)
                ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                plt.tight_layout()
                st.pyplot(fig1, width='stretch')
                plt.close(fig1)
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                if 'fig1' in locals():
                    plt.close(fig1)
        
        with col2:
            st.markdown("#### Contribution Magnitude")
            try:
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                abs_contrib = np.abs(exp_df['Contribution'].values)
                ax2.hist(abs_contrib, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
                ax2.set_xlabel('Absolute Contribution', fontsize=10)
                ax2.set_ylabel('Frequency', fontsize=10)
                ax2.set_title('Distribution of Contribution Magnitudes', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig2, width='stretch')
                plt.close(fig2)
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                if 'fig2' in locals():
                    plt.close(fig2)
        
        # Feature contributions table
        st.markdown("#### Feature Contributions Table")
        st.dataframe(exp_df, width='stretch')
            
    except ImportError:
        st.error("LIME library not installed. Please install it using: `pip install lime`")
    except Exception as e:
        error_msg = str(e)
        if not error_msg or error_msg == "1":
            error_msg = "Unknown error occurred. This might be due to model compatibility issues or data format problems."
        st.error(f"Error generating LIME explanations: {error_msg}")
        st.exception(e)

