"""
Page 3: Statistical Analysis
Interactive EDA with file upload and threshold filtering
"""

import streamlit as st
import pandas as pd
import numpy as np
import eda_functions as eda
import io


def show():
    st.title("📊 Statistical Analysis")
    st.markdown("---")
    
    # File upload section
    st.header("📁 Data Loading")
    
    uploaded_file = st.file_uploader(
        "Upload a CSV file (or use default dataset)",
        type=['csv'],
        help="Upload your own dataset or leave empty to use the default dataset"
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {data.shape}")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            data = None
    else:
        # Load default dataset
        try:
            data = eda.load_data("alzheimers_disease_data.csv")
            st.info(f"📊 Using default dataset. Shape: {data.shape}")
        except Exception as e:
            st.error(f"Error loading default dataset: {str(e)}")
            data = None
    
    if data is None:
        st.stop()
    
    # Store in session state
    st.session_state.current_data = data
    
    # Dataset Overview
    st.header("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Total Features", len(data.columns))
    with col3:
        st.metric("Missing Values", data.isnull().sum().sum())
    with col4:
        if 'Diagnosis' in data.columns:
            target_dist = data['Diagnosis'].value_counts()
            st.metric("Alzheimer Cases", f"{target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(data)*100:.1f}%)")
    
    # Threshold Filtering Section
    st.header("🔍 Data Filtering")
    st.markdown("Apply thresholds to filter the dataset before analysis")
    
    # Get numerical columns for filtering
    num_cols = eda.get_numerical_variables(data)
    
    # Create threshold inputs
    thresholds = {}
    filter_cols = st.multiselect(
        "Select columns to filter",
        options=num_cols,
        help="Select numerical columns to apply thresholds"
    )
    
    if filter_cols:
        threshold_cols = st.columns(min(len(filter_cols), 3))
        for idx, col in enumerate(filter_cols):
            with threshold_cols[idx % 3]:
                min_val = st.number_input(
                    f"{col} (min)",
                    value=float(data[col].min()),
                    min_value=float(data[col].min()),
                    max_value=float(data[col].max()),
                    key=f"min_{col}"
                )
                max_val = st.number_input(
                    f"{col} (max)",
                    value=float(data[col].max()),
                    min_value=float(data[col].min()),
                    max_value=float(data[col].max()),
                    key=f"max_{col}"
                )
                thresholds[col] = (min_val, max_val)
        
        # Apply filters
        if st.button("Apply Filters"):
            filtered_data = eda.filter_data_by_thresholds(data, thresholds)
            st.session_state.current_data = filtered_data
            st.success(f"✅ Filters applied! Filtered dataset shape: {filtered_data.shape}")
            data = filtered_data
    
    # Download filtered data
    if st.button("📥 Download Filtered Dataset"):
        csv = st.session_state.current_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="filtered_alzheimer_data.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # Analysis Sections
    st.header("📈 Analysis Sections")
    
    analysis_tabs = st.tabs([
        "Dataset Info",
        "Target Distribution",
        "Numerical Analysis",
        "Categorical Analysis",
        "Correlations",
        "Feature Importance",
        "Statistical Tests",
        "Risk Factors"
    ])
    
    with analysis_tabs[0]:
        st.subheader("Dataset Information")
        info = eda.get_dataset_info(data)
        st.write(f"**Shape:** {info['shape']}")
        st.write(f"**Columns:** {len(info['columns'])}")
        
        st.subheader("Data Types")
        st.dataframe(pd.DataFrame({
            'Column': list(info['dtypes'].keys()),
            'Data Type': list(info['dtypes'].values())
        }))
        
        st.subheader("Missing Values")
        null_df = pd.DataFrame({
            'Column': list(info['null_counts'].keys()),
            'Null Count': list(info['null_counts'].values())
        })
        st.dataframe(null_df[null_df['Null Count'] > 0] if null_df['Null Count'].sum() > 0 else null_df)
    
    with analysis_tabs[1]:
        st.subheader("Target Variable Distribution")
        if 'Diagnosis' in data.columns:
            class_counts, class_props = eda.get_target_distribution(data)
            st.dataframe(pd.DataFrame({
                'Class': class_counts.index,
                'Count': class_counts.values,
                'Percentage': class_props.values
            }))
            
            fig = eda.plot_target_distribution(data)
            st.pyplot(fig)
        else:
            st.warning("Target variable 'Diagnosis' not found in dataset")
    
    with analysis_tabs[2]:
        st.subheader("Numerical Variables Analysis")
        num_vars = eda.get_numerical_variables(data)
        
        if num_vars:
            st.write(f"**Numerical variables found:** {len(num_vars)}")
            st.dataframe(eda.get_descriptive_stats(data[num_vars]))
            
            # Plot selected variables
            selected_vars = st.multiselect(
                "Select variables to visualize (max 6)",
                options=num_vars,
                default=num_vars[:6] if len(num_vars) >= 6 else num_vars
            )
            
            if selected_vars and 'Diagnosis' in data.columns:
                fig = eda.plot_numerical_vs_target(data, selected_vars, max_vars=6)
                st.pyplot(fig)
        else:
            st.warning("No numerical variables found")
    
    with analysis_tabs[3]:
        st.subheader("Categorical Variables Analysis")
        cat_vars = eda.get_categorical_variables(data)
        
        if cat_vars:
            st.write(f"**Categorical variables found:** {len(cat_vars)}")
            
            # Plot selected variables
            selected_cat_vars = st.multiselect(
                "Select variables to visualize (max 6)",
                options=cat_vars,
                default=cat_vars[:6] if len(cat_vars) >= 6 else cat_vars
            )
            
            if selected_cat_vars and 'Diagnosis' in data.columns:
                fig = eda.plot_categorical_vs_target(data, selected_cat_vars, max_vars=6)
                st.pyplot(fig)
        else:
            st.warning("No categorical variables found")
    
    with analysis_tabs[4]:
        st.subheader("Correlation Analysis")
        num_vars = eda.get_numerical_variables(data)
        
        if num_vars and 'Diagnosis' in data.columns:
            # Correlation matrix
            fig = eda.plot_correlation_matrix(data, num_vars)
            st.pyplot(fig)
            
            # Target correlations
            st.subheader("Correlations with Target Variable")
            target_corr = eda.get_target_correlations(data, num_vars)
            st.dataframe(target_corr.to_frame('Correlation').head(15))
        else:
            st.warning("Cannot compute correlations - missing numerical variables or target")
    
    with analysis_tabs[5]:
        st.subheader("Feature Importance (Correlation-based)")
        num_vars = eda.get_numerical_variables(data)
        
        if num_vars and 'Diagnosis' in data.columns:
            corr_df = eda.get_feature_importance_correlation(data, num_vars)
            st.dataframe(corr_df.head(15))
            
            fig = eda.plot_feature_importance_correlation(corr_df)
            st.pyplot(fig)
        else:
            st.warning("Cannot compute feature importance - missing numerical variables or target")
    
    with analysis_tabs[6]:
        st.subheader("Statistical Significance Tests")
        num_vars = eda.get_numerical_variables(data)
        
        if num_vars and 'Diagnosis' in data.columns:
            stats_df = eda.statistical_tests_numerical(data, num_vars)
            st.dataframe(stats_df)
            
            significant = stats_df[stats_df['Significant (p<0.05)'] == 'Yes']
            if len(significant) > 0:
                st.success(f"✅ {len(significant)} features are statistically significant (p < 0.05)")
                st.dataframe(significant[['Feature', 'P-value', 'Difference']])
        else:
            st.warning("Cannot perform statistical tests - missing numerical variables or target")
        
        # Chi-square tests
        st.subheader("Chi-Square Tests (Categorical)")
        cat_vars = eda.get_categorical_variables(data)
        
        if cat_vars and 'Diagnosis' in data.columns:
            chi2_df = eda.chi_square_tests_categorical(data, cat_vars)
            st.dataframe(chi2_df)
            
            significant_cat = chi2_df[chi2_df['Significant (p<0.05)'] == 'Yes']
            if len(significant_cat) > 0:
                st.success(f"✅ {len(significant_cat)} categorical features are statistically significant (p < 0.05)")
                st.dataframe(significant_cat[['Feature', 'P-value', 'Chi-square']])
    
    with analysis_tabs[7]:
        st.subheader("Risk Factor Analysis")
        binary_risk_factors = [
            'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 
            'Depression', 'HeadInjury', 'Hypertension', 'Smoking',
            'MemoryComplaints', 'BehavioralProblems', 'Confusion', 
            'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 
            'Forgetfulness'
        ]
        
        if 'Diagnosis' in data.columns:
            risk_df = eda.risk_factor_analysis(data, binary_risk_factors)
            st.dataframe(risk_df)
            
            fig = eda.plot_risk_factors(risk_df)
            st.pyplot(fig)
        else:
            st.warning("Cannot perform risk factor analysis - missing target variable")

