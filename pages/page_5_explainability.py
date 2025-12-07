"""
Page 5: Explainability
Understand model predictions using SHAP, ELI5, and LIME
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
        ["SHAP", "ELI5", "LIME"],
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
        elif explainability_method == "ELI5":
            show_eli5_global(model, data_dict, model_type)
        elif explainability_method == "LIME":
            st.info("LIME is primarily designed for local explanations. Please select 'Local' analysis type for LIME.")
    
    else:  # Local explanations
        st.header("🔬 Local Explainability")
        st.markdown("""
        **What this shows**: Why the model made a specific prediction for an individual patient. 
        This helps you understand which of this patient's specific factors contributed to their risk assessment.
        
        **Clinical Use**: Use this to:
        - Explain predictions to patients and families
        - Identify which specific factors are driving a patient's risk
        - Validate that the model's reasoning aligns with clinical assessment
        """)
        
        # Select sample to explain
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        y_pred = st.session_state.model_results['y_pred']
        y_proba = st.session_state.model_results['y_proba']
        
        # Find interesting cases
        tp_indices = np.where((y_pred == 1) & (y_test == 1))[0]
        tn_indices = np.where((y_pred == 0) & (y_test == 0))[0]
        fp_indices = np.where((y_pred == 1) & (y_test == 0))[0]
        fn_indices = np.where((y_pred == 0) & (y_test == 1))[0]
        
        case_options = {}
        if len(tp_indices) > 0:
            case_options[f"True Positive (Sample {tp_indices[0]})"] = tp_indices[0]
        if len(tn_indices) > 0:
            case_options[f"True Negative (Sample {tn_indices[0]})"] = tn_indices[0]
        if len(fp_indices) > 0:
            case_options[f"False Positive (Sample {fp_indices[0]})"] = fp_indices[0]
        if len(fn_indices) > 0:
            case_options[f"False Negative (Sample {fn_indices[0]})"] = fn_indices[0]
        
        if case_options:
            selected_case = st.selectbox(
                "Select a case to explain",
                options=list(case_options.keys())
            )
            sample_idx = case_options[selected_case]
            
            st.subheader(f"Case Details: {selected_case}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Actual", "Alzheimer" if y_test.iloc[sample_idx] == 1 else "No Alzheimer")
            with col2:
                st.metric("Predicted", "Alzheimer" if y_pred[sample_idx] == 1 else "No Alzheimer")
            with col3:
                st.metric("Probability", f"{y_proba[sample_idx]:.4f}")
            
            # Show explanation
            if explainability_method == "SHAP":
                show_shap_local(model, data_dict, sample_idx, model_type)
            elif explainability_method == "ELI5":
                show_eli5_local(model, data_dict, sample_idx, model_type)
            elif explainability_method == "LIME":
                show_lime_local(model, data_dict, sample_idx, model_type)
        else:
            st.warning("No test cases available for explanation.")


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
            st.markdown("#### Dependence Plots")
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


def show_eli5_global(model, data_dict, model_type):
    """Show ELI5 global explanations."""
    try:
        import eli5
        
        st.subheader("ELI5 Global Explanations")
        st.markdown("""
        ELI5 (Explain Like I'm 5) provides intuitive feature importance explanations.
        """)
        
        feature_names = data_dict['feature_names']
        
        # Get feature weights
        if hasattr(model, 'feature_importances_'):
            weights = model.feature_importances_
        elif hasattr(model, 'coef_'):
            weights = np.abs(model.coef_[0])
        else:
            st.warning("Model type not fully supported for ELI5 global explanations.")
            return
        
        weights_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': weights
        }).sort_values('Importance', ascending=False)
        
        # Calculate variability in feature values from training data for better visualization
        X_train = data_dict.get('X_train', None)
        if X_train is not None:
            # Calculate coefficient of variation (std/mean) for each feature
            feature_variability = []
            for feat in feature_names:
                if feat in X_train.columns:
                    feat_values = X_train[feat].values
                    if np.std(feat_values) > 0:
                        cv = np.std(feat_values) / (np.abs(np.mean(feat_values)) + 1e-10)
                    else:
                        cv = 0
                    feature_variability.append(cv)
                else:
                    feature_variability.append(0)
            
            # Combine importance and variability
            weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
            var_norm = np.array(feature_variability)
            if var_norm.max() > 0:
                var_norm = (var_norm - var_norm.min()) / (var_norm.max() - var_norm.min() + 1e-10)
            combined_score = 0.7 * weights_norm + 0.3 * var_norm
            # Select top features by combined score
            top_indices = np.argsort(combined_score)[-10:][::-1]
            top_features_selected = weights_df.iloc[top_indices].sort_values('Importance', ascending=False)
        else:
            top_features_selected = weights_df.head(10)
        
        # Visualizations side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Feature Importance Bar Chart")
            import matplotlib.pyplot as plt
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            ax1.barh(range(len(top_features_selected)), top_features_selected['Importance'])
            ax1.set_yticks(range(len(top_features_selected)))
            ax1.set_yticklabels(top_features_selected['Feature'], fontsize=9)
            ax1.set_xlabel('Importance', fontsize=10)
            ax1.set_title('Top 10 Features (High Variability)', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig1, width='stretch')
        
        with col2:
            st.markdown("#### Feature Importance Distribution")
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.hist(weights, bins=20, edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Importance Value', fontsize=10)
            ax2.set_ylabel('Frequency', fontsize=10)
            ax2.set_title('Distribution of Feature Importance', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig2, width='stretch')
        
        # Cumulative importance plot
        st.markdown("#### Cumulative Feature Importance")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sorted_weights = np.sort(weights)[::-1]
        cumulative = np.cumsum(sorted_weights)
        ax3.plot(range(len(cumulative)), cumulative / cumulative[-1] * 100, marker='o', markersize=4)
        ax3.set_xlabel('Number of Features', fontsize=10)
        ax3.set_ylabel('Cumulative Importance (%)', fontsize=10)
        ax3.set_title('Cumulative Feature Importance', fontsize=11)
        ax3.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3, width='stretch')
        
        # Table
        st.markdown("#### Feature Importance Table")
        st.dataframe(weights_df.head(15), width='stretch')
        
    except ImportError:
        st.error("ELI5 library not installed. Please install it using: `pip install eli5`")
    except Exception as e:
        st.error(f"Error generating ELI5 explanations: {str(e)}")


def show_shap_local(model, data_dict, sample_idx, model_type):
    """Show SHAP local explanations."""
    try:
        import shap
        
        st.subheader("SHAP Local Explanation")
        st.markdown("""
        **What this shows**: How each of this patient's specific clinical factors contributed 
        to their individual risk prediction.
        """)
        
        X_test = data_dict['X_test']
        X_sample = X_test.iloc[sample_idx:sample_idx+1]
        
        with st.spinner("Calculating SHAP values..."):
            if model_type in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X_sample)
            
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
                plt.ioff()  # Turn off interactive mode
                try:
                    shap.waterfall_plot(shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0], show=False)
                    fig1 = plt.gcf()  # Get current figure
                    st.pyplot(fig1, width='stretch')
                    plt.close(fig1)
                except:
                    # Fallback
                    fig1, ax = plt.subplots(figsize=(6, 5))
                    top_indices = np.argsort(np.abs(shap_values_class1))[-10:][::-1]
                    ax.barh(range(len(top_indices)), shap_values_class1[top_indices])
                    ax.set_yticks(range(len(top_indices)))
                    ax.set_yticklabels([X_test.columns[i] for i in top_indices], fontsize=9)
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
                
                import matplotlib.pyplot as plt
                contributions_df = pd.DataFrame({
                    'Feature': X_test.columns,
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
            
            # Feature contributions table
            contributions_df = pd.DataFrame({
                'Feature': X_test.columns,
                'SHAP Value': shap_values_class1
            }).sort_values('SHAP Value', key=abs, ascending=False)
            
            st.markdown("#### Feature Contributions Table")
            st.dataframe(contributions_df.head(15), width='stretch')
            
    except ImportError:
        st.error("SHAP library not installed. Please install it using: `pip install shap`")
    except Exception as e:
        st.error(f"Error generating SHAP local explanations: {str(e)}")


def show_eli5_local(model, data_dict, sample_idx, model_type):
    """Show ELI5 local explanations."""
    try:
        import eli5
        
        st.subheader("ELI5 Local Explanation")
        
        X_test = data_dict['X_test']
        X_sample = X_test.iloc[sample_idx]
        feature_names = data_dict['feature_names']
        
        # Get explanation
        explanation = eli5.explain_prediction(
            model,
            X_sample,
            feature_names=feature_names,
            top=15
        )
        
        # Try to extract feature weights from explanation
        try:
            # Parse explanation to get feature contributions
            exp_html = eli5.format_as_dict(explanation)
            if 'feature_importances' in exp_html:
                weights = exp_html['feature_importances']
                features = [w['feature'] for w in weights]
                values = [w['weight'] for w in weights]
                
                # Create visualizations side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Feature Contributions")
                    import matplotlib.pyplot as plt
                    fig1, ax1 = plt.subplots(figsize=(6, 5))
                    top_10_idx = np.argsort(np.abs(values))[-10:][::-1]
                    top_features = [features[i] for i in top_10_idx]
                    top_values = [values[i] for i in top_10_idx]
                    colors = ['red' if v < 0 else 'green' for v in top_values]
                    ax1.barh(range(len(top_features)), top_values, color=colors, alpha=0.7)
                    ax1.set_yticks(range(len(top_features)))
                    ax1.set_yticklabels(top_features, fontsize=9)
                    ax1.set_xlabel('Contribution', fontsize=10)
                    ax1.set_title('Top 10 Feature Contributions', fontsize=11)
                    ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                    plt.tight_layout()
                    st.pyplot(fig1, width='stretch')
                
                with col2:
                    st.markdown("#### Text Explanation")
                    st.text(str(explanation))
            else:
                st.markdown("#### Feature Contributions")
                st.text(str(explanation))
        except:
            # Fallback to text explanation
            st.markdown("#### Feature Contributions")
            st.text(str(explanation))
        
    except ImportError:
        st.error("ELI5 library not installed. Please install it using: `pip install eli5`")
    except Exception as e:
        st.error(f"Error generating ELI5 local explanations: {str(e)}")


def show_lime_local(model, data_dict, sample_idx, model_type):
    """Show LIME local explanations."""
    try:
        import lime
        import lime.lime_tabular
        
        st.subheader("LIME Local Explanation")
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        X_sample = X_test.iloc[sample_idx]
        feature_names = data_dict['feature_names']
        
        with st.spinner("Creating LIME explainer..."):
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=feature_names,
                class_names=['No Alzheimer', 'Alzheimer'],
                mode='classification',
                training_labels=y_train.values
            )
            
            explanation = explainer.explain_instance(
                X_sample.values,
                model.predict_proba,
                num_features=15,
                top_labels=1
            )
            
            # Get explanation as list
            exp_list = explanation.as_list(label=1)
            
            # Create DataFrame
            exp_df = pd.DataFrame(exp_list, columns=['Feature', 'Contribution'])
            exp_df = exp_df.sort_values('Contribution', key=abs, ascending=False)
            
            # Visualizations side by side
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Feature Contributions Bar Chart")
                import matplotlib.pyplot as plt
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
            
            with col2:
                st.markdown("#### Contribution Magnitude")
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                abs_contrib = np.abs(exp_df['Contribution'].values)
                ax2.hist(abs_contrib, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
                ax2.set_xlabel('Absolute Contribution', fontsize=10)
                ax2.set_ylabel('Frequency', fontsize=10)
                ax2.set_title('Distribution of Contribution Magnitudes', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig2, width='stretch')
            
            # Feature contributions table
            st.markdown("#### Feature Contributions Table")
            st.dataframe(exp_df, width='stretch')
            
    except ImportError:
        st.error("LIME library not installed. Please install it using: `pip install lime`")
    except Exception as e:
        st.error(f"Error generating LIME explanations: {str(e)}")

