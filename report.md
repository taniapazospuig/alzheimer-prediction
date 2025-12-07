# Alzheimer Disease Prediction: Machine Learning Approach

## Problem Statement

Alzheimer's disease is a progressive neurodegenerative disorder that affects millions of people worldwide. Early detection and intervention are critical for managing the disease and potentially slowing its progression. However, identifying patients at risk of developing Alzheimer's disease presents significant challenges for clinicians due to the complexity of contributing factors and the subtle nature of early symptoms.

The primary goal of this project is to develop a machine learning system that, given a patient's medical condition and clinical features, can predict with high confidence whether a patient will develop Alzheimer's disease or not. This is a binary classification problem where the model must distinguish between patients who will be diagnosed with Alzheimer's (positive class) and those who will not (negative class).

The clinical context is critical: missing a true case of Alzheimer's disease (false negative) has severe consequences, as it means:
- A patient with Alzheimer's is not identified early
- They miss critical early intervention opportunities
- Disease progression continues unchecked, potentially leading to irreversible cognitive decline

Therefore, the system prioritizes **high sensitivity (recall)** over precision, aiming to identify as many true Alzheimer's cases as possible, even if this results in some false positives that can be ruled out through further clinical testing.

---

## Dataset Overview

### Data Source
The dataset used in this project contains clinical and demographic information for patients, with the target variable being the diagnosis of Alzheimer's disease (binary classification: 0 = No Alzheimer's, 1 = Alzheimer's).

### Dataset Characteristics
The dataset includes a comprehensive set of features covering:

**Demographic Information:**
- Age, Gender, Ethnicity, Education Level

**Lifestyle Factors:**
- BMI (Body Mass Index)
- Smoking status
- Alcohol consumption
- Physical activity levels
- Diet quality
- Sleep quality

**Medical History:**
- Family history of Alzheimer's
- Cardiovascular disease
- Diabetes
- Depression
- Head injury history
- Hypertension

**Clinical Measurements:**
- Systolic and Diastolic Blood Pressure
- Cholesterol levels (Total, LDL, HDL, Triglycerides)

**Cognitive Assessments:**
- MMSE (Mini-Mental State Examination) scores
- Functional Assessment scores
- Memory complaints
- Behavioral problems
- Activities of Daily Living (ADL) scores
- Confusion, Disorientation
- Personality changes
- Difficulty completing tasks
- Forgetfulness

### Exploratory Data Analysis
The dataset exhibits class imbalance, which is typical for medical datasets where disease cases are less common than non-disease cases. This imbalance is addressed through class weighting and threshold optimization techniques during model training.

Key observations from exploratory analysis:
- Multiple clinical and demographic features that may contribute to Alzheimer's risk
- Presence of both continuous (e.g., MMSE scores, blood pressure) and categorical (e.g., smoking status, family history) variables
- Features that have direct clinical meaning, preserving interpretability for healthcare providers

---

## Business Questions and Objectives

### Key Business Questions
1. **Can machine learning models accurately predict Alzheimer's disease risk based on clinical and demographic features?**
   - This addresses the core predictive capability of the system.

2. **Which clinical features are most important in predicting Alzheimer's disease?**
   - Understanding feature importance helps clinicians focus on the most relevant risk factors.

3. **Can we achieve high sensitivity (recall) to minimize false negatives while maintaining acceptable precision?**
   - This is critical for clinical deployment, as missing true cases has severe consequences.

4. **How interpretable are the model predictions for clinical decision support?**
   - Clinicians need to understand and trust the model's reasoning to integrate it into clinical workflows.

### Project Objectives
1. **Develop Multiple ML Models**: Implement and compare four different machine learning algorithms:
   - Logistic Regression (interpretable linear model)
   - Decision Tree (rule-based, highly interpretable)
   - Random Forest (ensemble method for robustness)
   - Gradient Boosting (high-performance ensemble method)

2. **Prioritize Clinical Applicability**: 
   - Maintain high sensitivity to minimize false negatives
   - Preserve clinical interpretability of features
   - Enable explainability for individual predictions

3. **Comprehensive Evaluation**: 
   - Use multiple metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC)
   - Optimize decision thresholds for clinical objectives
   - Analyze confusion matrices to understand model behavior

4. **Explainability Integration**: 
   - Implement SHAP, ELI5, and LIME for model interpretation
   - Provide both global (model-wide) and local (individual prediction) explanations

---

## Methodology

### Data Preprocessing
The preprocessing pipeline performs minimal, essential transformations to preserve clinical interpretability:

- **Missing Value Handling**: Identification and appropriate handling of missing values
- **Train-Test Split**: 80-20 split with stratification to maintain class distribution across splits
- **Feature Scaling**: RobustScaler applied only for models that require it (Logistic Regression)
- **No Complex Feature Engineering**: Deliberately avoided extensive feature transformations to preserve the clinical meaning of variables

### Model Selection and Rationale

**1. Logistic Regression**
- **Rationale**: Highly interpretable linear model with directly interpretable coefficients
- **Implementation**: Uses maximum likelihood estimation with RobustScaler for feature normalization
- **Advantages**: Fast, interpretable, provides calibrated probabilities
- **Limitations**: Assumes linear relationships

**2. Decision Tree Classifier**
- **Rationale**: Rule-based model that produces human-readable decision paths
- **Implementation**: Recursive partitioning with Gini impurity, limited depth (max_depth=10) to prevent overfitting
- **Advantages**: Highly interpretable, captures non-linear relationships, no distribution assumptions
- **Limitations**: Prone to overfitting if not regularized

**3. Random Forest Classifier**
- **Rationale**: Ensemble method that reduces overfitting while maintaining interpretability through feature importance
- **Implementation**: 100 trees with bootstrap sampling and random feature selection at each split
- **Advantages**: Robust, handles non-linear relationships, provides feature importance rankings
- **Limitations**: Less directly interpretable than single tree, but explainable via feature importance

**4. Gradient Boosting Classifier**
- **Rationale**: Sequential ensemble method that often achieves best predictive performance
- **Implementation**: Iteratively adds weak learners (shallow trees) to correct previous errors using gradient descent
- **Advantages**: High predictive performance, captures complex relationships
- **Limitations**: More prone to overfitting, requires careful tuning

### Class Imbalance Handling
- **Class Weights**: Used `class_weight='balanced'` to automatically adjust loss function, penalizing misclassification of minority class (Alzheimer's cases) more heavily
- **Threshold Optimization**: Explored precision-recall curves to find optimal probability thresholds that meet clinical recall targets (e.g., ≥90% sensitivity) rather than using default 0.5 cutoff
- **Rationale**: Preserves original data distribution while ensuring focus on correctly identifying Alzheimer's cases

### Evaluation Strategy
**Metrics Used:**
- **Accuracy**: Overall correctness
- **Precision**: Proportion of predicted Alzheimer's cases that are actually Alzheimer's
- **Recall (Sensitivity)**: Proportion of actual Alzheimer's cases correctly identified (PRIORITIZED)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (class separation ability)
- **PR-AUC**: Area under Precision-Recall curve (better for imbalanced data)

**Confusion Matrix Analysis:**
- Explicit tracking of True Positives, True Negatives, False Positives, and False Negatives
- Special attention to False Negatives (missed Alzheimer's cases) as the critical metric

**Threshold Optimization:**
- Generated precision-recall curves across all possible thresholds
- Used F1 optimization, Youden's J statistic, or clinical recall targets to select optimal thresholds
- Balanced trade-off between precision and recall based on clinical needs

### Explainability Approach
Multiple explainability techniques ensure clinical acceptance:

1. **SHAP (SHapley Additive exPlanations)**: Provides unified, theoretically grounded feature importance values based on game theory. Shows both global (overall model) and local (individual prediction) explanations.

2. **ELI5 (Explain Like I'm 5)**: Offers intuitive explanations for model predictions, showing feature weights and contributions in an accessible format.

3. **LIME (Local Interpretable Model-agnostic Explanations)**: Creates local surrogate models to explain individual predictions by approximating the complex model with a simpler, interpretable model in the neighborhood of each prediction.

These methods allow clinicians to:
- Understand which features drive model decisions globally
- Explain specific predictions for individual patients
- Validate that model reasoning aligns with medical knowledge

### Technical Implementation
- **Hyperparameter Tuning**: Optional GridSearchCV with cross-validation for optimal model parameters
- **Model Comparison**: All four models implemented to compare performance and interpretability trade-offs
- **Streamlit Application**: Interactive web application for model training, evaluation, and prediction with explainability visualization

---

## Conclusions

### Key Findings
1. **Model Performance**: The implemented machine learning models demonstrate strong predictive capability for Alzheimer's disease risk assessment. Gradient Boosting and Random Forest typically achieve the highest overall performance, while Logistic Regression and Decision Trees provide superior interpretability.

2. **Feature Importance**: Clinical features such as MMSE scores, cognitive assessments, and medical history factors emerge as key predictors, aligning with established medical knowledge about Alzheimer's risk factors.

3. **Clinical Applicability**: By prioritizing high sensitivity (recall) through class weighting and threshold optimization, the models can achieve clinical recall targets (≥90% sensitivity) while maintaining acceptable precision, making them suitable for early screening applications.

4. **Interpretability**: The combination of inherently interpretable models (Logistic Regression, Decision Tree) and post-hoc explainability tools (SHAP, ELI5, LIME) provides comprehensive insights that clinicians can understand and validate.

### Clinical Implications
- **Early Detection**: The system can support early identification of patients at risk, enabling timely intervention and potentially slowing disease progression.
- **Clinical Decision Support**: Explainable predictions allow clinicians to understand model reasoning and integrate insights into their decision-making process.
- **Resource Optimization**: High sensitivity reduces false negatives, ensuring that patients who need further evaluation are not missed, while threshold optimization helps balance resource allocation.

### Limitations and Future Work
- **Dataset Specificity**: Models are trained on specific datasets and may require external validation before clinical deployment across different populations.
- **Temporal Aspects**: Current implementation uses cross-sectional data; future work could incorporate longitudinal data to capture disease progression over time.
- **Clinical Validation**: Real-world clinical validation studies are needed to assess model performance in actual clinical settings.

### Summary
This project successfully develops a machine learning system for Alzheimer's disease prediction that prioritizes clinical applicability and interpretability while maintaining strong predictive performance. By avoiding extensive feature engineering, the system preserves the clinical meaning of variables, enabling healthcare providers to understand and trust model predictions. The combination of multiple models and explainability techniques provides a robust, transparent system that can support clinical decision-making while maintaining the flexibility to adapt to different clinical needs through threshold optimization.

---

*This report documents the technical approach, methodology, and findings of the Alzheimer Disease Prediction project. The implementation prioritizes clinical applicability, interpretability, and high sensitivity to support early detection and intervention strategies.*
