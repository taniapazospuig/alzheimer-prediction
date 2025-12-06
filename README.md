# Alzheimer Disease Prediction - Streamlit Application

A comprehensive Streamlit application for predicting Alzheimer's disease using machine learning models, with a focus on explainability and clinical applicability.

## Features

- **6 Interactive Pages**:
  1. **Introduction**: Overview of the application
  2. **Technical Description**: Complete project documentation
  3. **Statistical Analysis**: Interactive EDA with data filtering
  4. **Model Training**: Train and evaluate 4 different ML models
  5. **Explainability**: Understand predictions using SHAP, ELI5, and LIME
  6. **Prediction**: Make predictions for individual patients

- **4 Machine Learning Models**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting

- **Explainability Methods**:
  - SHAP (SHapley Additive exPlanations)
  - ELI5 (Explain Like I'm 5)
  - LIME (Local Interpretable Model-agnostic Explanations)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure you have the dataset file `alzheimers_disease_data.csv` in the project root directory.

## Running the Application

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage

1. **Start with Introduction**: Understand the application structure and features.

2. **Explore Data**: Use the Statistical Analysis page to:
   - Upload your own dataset or use the default
   - Apply filters and thresholds
   - View statistical summaries and visualizations
   - Download filtered datasets

3. **Train Models**: Go to Model Training page to:
   - Select a model type
   - Upload data or use default
   - Enable hyperparameter tuning (optional)
   - View comprehensive performance metrics

4. **Understand Predictions**: Use Explainability page to:
   - View global feature importance
   - Analyze individual predictions
   - Compare different explainability methods

5. **Make Predictions**: Use Prediction page to:
   - Input patient features
   - Get instant predictions with probabilities
   - View feature contributions

## Project Structure

```
alzheimer-prediction/
├── app.py                      # Main Streamlit application
├── eda_functions.py           # EDA utility functions
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── utils.py               # Shared utilities
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   └── gradient_boosting.py
├── pages/                      # Streamlit pages
│   ├── __init__.py
│   ├── page_1_introduction.py
│   ├── page_2_technical_description.py
│   ├── page_3_statistical_analysis.py
│   ├── page_4_model_training.py
│   ├── page_5_explainability.py
│   └── page_6_prediction.py
├── alzheimers_disease_data.csv # Default dataset
├── project_summary.txt         # Technical documentation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full list of dependencies

## Notes

- The application prioritizes **high sensitivity (recall)** to minimize false negatives
- Models are trained with class weights to handle imbalanced data
- Threshold optimization is available to meet clinical recall targets
- All models support both global and local explainability

## License

This project is for educational and research purposes.

