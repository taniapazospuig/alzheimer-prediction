"""
EDA Functions for Alzheimer Disease Prediction
Extracted from eda.ipynb for use in Streamlit application
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)


def get_dataset_info(data):
    """Get basic dataset information."""
    info = {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.to_dict(),
        'null_counts': data.isnull().sum().to_dict(),
        'memory_usage': data.memory_usage(deep=True).sum()
    }
    return info


def get_descriptive_stats(data):
    """Get descriptive statistics for numerical variables."""
    return data.describe()


def get_target_distribution(data, target_col='Diagnosis'):
    """Get target variable distribution."""
    class_counts = data[target_col].value_counts().rename({0: "No Alzheimer", 1: "Alzheimer"})
    class_props = (class_counts / len(data) * 100).round(2)
    return class_counts, class_props


def plot_target_distribution(data, target_col='Diagnosis'):
    """Plot target variable distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))
    class_counts = data[target_col].value_counts().rename({0: "No Alzheimer", 1: "Alzheimer"})
    ax = sns.barplot(x=class_counts.index, y=class_counts.values, 
                     hue=class_counts.index, palette="viridis", legend=False, ax=ax)
    ax.set_title("Diagnosis Distribution")
    ax.set_ylabel("Number of Patients")
    for p, c in zip(ax.patches, class_counts.values):
        ax.annotate(f"{c}\n({c/len(data)*100:.1f}%)", 
                   (p.get_x() + p.get_width()/2, p.get_height()),
                   ha="center", va="bottom")
    plt.tight_layout()
    return fig


def get_numerical_variables(data, exclude_cols=['PatientID', 'Diagnosis', 'DoctorInCharge']):
    """Get list of numerical variables."""
    num_variables = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 
                     'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
                     'MMSE', 'FunctionalAssessment', 'ADL', 'AlcoholConsumption',
                     'PhysicalActivity', 'DietQuality', 'SleepQuality']
    return [col for col in num_variables if col in data.columns and col not in exclude_cols]


def get_categorical_variables(data):
    """Get list of categorical variables."""
    binary_variables = ['Gender', 'Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease', 
                        'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 'MemoryComplaints',
                        'BehavioralProblems', 'Confusion', 'Disorientation', 'PersonalityChanges',
                        'DifficultyCompletingTasks', 'Forgetfulness']
    multi_cat_variables = ['Ethnicity', 'EducationLevel']
    categorical_variables = binary_variables + multi_cat_variables
    return [col for col in categorical_variables if col in data.columns]


def plot_numerical_vs_target(data, num_variables, target_col='Diagnosis', max_vars=6):
    """Plot numerical variables vs target (limited to max_vars for performance)."""
    num_vars = min(len(num_variables), max_vars)
    cols = 2
    rows = num_vars
    
    fig, axs = plt.subplots(rows, cols, figsize=(15, 4*rows))
    if rows == 1:
        axs = axs.reshape(1, -1)
    axs = axs.flatten()
    
    for i, column in enumerate(num_variables[:max_vars]):
        # Box plot
        sns.boxplot(x=target_col, y=column, data=data, ax=axs[i*2])
        axs[i*2].set_title(f'Box plot of {column} by Diagnosis')
        
        # Distribution plot
        sns.histplot(data=data, x=column, hue=target_col, kde=True, 
                    element='step', ax=axs[i*2+1])
        axs[i*2+1].set_title(f'Distribution of {column} by Diagnosis')
    
    plt.tight_layout()
    return fig


def plot_categorical_vs_target(data, categorical_variables, target_col='Diagnosis', max_vars=6):
    """Plot categorical variables vs target (limited to max_vars for performance)."""
    num_vars = min(len(categorical_variables), max_vars)
    cols = 2
    rows = num_vars
    
    fig, axs = plt.subplots(rows, cols, figsize=(15, 4*rows))
    if rows == 1:
        axs = axs.reshape(1, -1)
    axs = axs.flatten()
    
    for i, column in enumerate(categorical_variables[:max_vars]):
        # Count plot
        sns.countplot(x=column, data=data, ax=axs[i*2])
        axs[i*2].set_title(f'Count plot of {column}')
        axs[i*2].tick_params(axis='x', rotation=45)
        
        # Distribution by diagnosis
        sns.countplot(x=column, hue=target_col, data=data, ax=axs[i*2+1])
        axs[i*2+1].set_title(f'Distribution of {column} by Diagnosis')
        axs[i*2+1].tick_params(axis='x', rotation=45)
        axs[i*2+1].legend(title='Diagnosis', labels=['No Alzheimer', 'Alzheimer'])
    
    plt.tight_layout()
    return fig


def get_correlation_matrix(data, num_variables, target_col='Diagnosis'):
    """Get correlation matrix for numerical variables."""
    corr_variables = num_variables + [target_col]
    corr_variables = [col for col in corr_variables if col in data.columns]
    return data[corr_variables].corr()


def plot_correlation_matrix(data, num_variables, target_col='Diagnosis'):
    """Plot correlation matrix."""
    corr_matrix = get_correlation_matrix(data, num_variables, target_col)
    
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix of Numerical Variables', fontsize=14, pad=20)
    plt.tight_layout()
    return fig


def get_target_correlations(data, num_variables, target_col='Diagnosis'):
    """Get correlations with target variable."""
    corr_matrix = get_correlation_matrix(data, num_variables, target_col)
    target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    return target_corr


def detect_outliers_iqr(data, numeric_columns):
    """Detect outliers using IQR method."""
    outlier_summary = []
    
    for col in numeric_columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(data)) * 100
        
        outlier_summary.append({
            'Feature': col,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Outlier Count': outlier_count,
            'Outlier %': f"{outlier_pct:.2f}%"
        })
    
    return pd.DataFrame(outlier_summary)


def get_feature_importance_correlation(data, numeric_cols, target_col='Diagnosis'):
    """Get feature importance based on correlation with target."""
    corr_with_target = {}
    for col in numeric_cols:
        if col != target_col:
            corr_val = data[col].corr(data[target_col])
            corr_with_target[col] = corr_val
    
    corr_df = pd.DataFrame({
        'Feature': list(corr_with_target.keys()),
        'Correlation': list(corr_with_target.values())
    })
    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    return corr_df


def plot_feature_importance_correlation(corr_df, top_n=15):
    """Plot feature importance based on correlation."""
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = corr_df.head(top_n)
    colors = ['red' if x < 0 else 'green' for x in top_features['Correlation']]
    bars = ax.barh(range(len(top_features)), top_features['Correlation'], 
                   color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'])
    ax.set_xlabel('Correlation with Diagnosis', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Feature Importance: Correlation with Diagnosis', fontsize=14, pad=20)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['Correlation'], i, f"{row['Correlation']:.3f}", 
               va='center', ha='left' if row['Correlation'] > 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    return fig


def statistical_tests_numerical(data, numeric_features, target_col='Diagnosis'):
    """Perform t-tests for numerical features."""
    statistical_results = []
    
    for feature in numeric_features:
        if feature != target_col:
            group_0 = data[data[target_col] == 0][feature]
            group_1 = data[data[target_col] == 1][feature]
            
            t_stat, p_value = stats.ttest_ind(group_0, group_1)
            mean_0 = group_0.mean()
            mean_1 = group_1.mean()
            
            statistical_results.append({
                'Feature': feature,
                'Mean (No Alzheimer)': f"{mean_0:.3f}",
                'Mean (Alzheimer)': f"{mean_1:.3f}",
                'Difference': f"{mean_1 - mean_0:.3f}",
                'T-statistic': f"{t_stat:.3f}",
                'P-value': f"{p_value:.6f}",
                'Significant (p<0.05)': 'Yes' if p_value < 0.05 else 'No'
            })
    
    return pd.DataFrame(statistical_results)


def chi_square_tests_categorical(data, categorical_features, target_col='Diagnosis'):
    """Perform chi-square tests for categorical features."""
    chi2_results = []
    
    for feature in categorical_features:
        if feature != target_col:
            contingency_table = pd.crosstab(data[feature], data[target_col])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            chi2_results.append({
                'Feature': feature,
                'Chi-square': f"{chi2:.3f}",
                'P-value': f"{p_value:.6f}",
                'Degrees of Freedom': dof,
                'Significant (p<0.05)': 'Yes' if p_value < 0.05 else 'No'
            })
    
    return pd.DataFrame(chi2_results)


def risk_factor_analysis(data, binary_risk_factors, target_col='Diagnosis'):
    """Analyze risk factors."""
    risk_analysis = []
    
    for factor in binary_risk_factors:
        if factor in data.columns:
            no_alz_with = data[(data[target_col] == 0) & (data[factor] == 1)].shape[0]
            no_alz_total = data[data[target_col] == 0].shape[0]
            alz_with = data[(data[target_col] == 1) & (data[factor] == 1)].shape[0]
            alz_total = data[data[target_col] == 1].shape[0]
            
            prev_no_alz = (no_alz_with / no_alz_total) * 100 if no_alz_total > 0 else 0
            prev_alz = (alz_with / alz_total) * 100 if alz_total > 0 else 0
            
            risk_ratio = prev_alz / prev_no_alz if prev_no_alz > 0 else np.inf
            
            risk_analysis.append({
                'Risk Factor': factor,
                'Prevalence (No Alzheimer)': f"{prev_no_alz:.2f}%",
                'Prevalence (Alzheimer)': f"{prev_alz:.2f}%",
                'Risk Ratio': f"{risk_ratio:.2f}" if risk_ratio != np.inf else "N/A",
                'Difference': f"{prev_alz - prev_no_alz:.2f}%"
            })
    
    risk_df = pd.DataFrame(risk_analysis)
    risk_df['Risk_Ratio_Numeric'] = pd.to_numeric(
        risk_df['Risk Ratio'].replace('N/A', '0'), errors='coerce'
    )
    risk_df = risk_df.sort_values('Risk_Ratio_Numeric', ascending=False)
    return risk_df


def plot_risk_factors(risk_df):
    """Plot risk factor analysis."""
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(risk_df))
    risk_ratios = risk_df['Risk_Ratio_Numeric'].values
    colors = ['red' if x > 1.5 else 'orange' if x > 1.0 else 'green' for x in risk_ratios]
    
    bars = ax.barh(y_pos, risk_ratios, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(risk_df['Risk Factor'])
    ax.set_xlabel('Risk Ratio', fontsize=12)
    ax.set_title('Risk Ratio Analysis: Prevalence in Alzheimer vs No Alzheimer', 
                 fontsize=14, pad=20)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, label='No difference (RR=1)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(risk_df.iterrows()):
        if row['Risk Ratio'] != 'N/A':
            ax.text(risk_ratios[i], i, f"{row['Risk Ratio']}", 
                   va='center', ha='left', fontsize=9)
    
    plt.tight_layout()
    return fig


def filter_data_by_thresholds(data, thresholds):
    """
    Filter data based on threshold values.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    thresholds : dict
        Dictionary with column names as keys and tuples (min, max) as values
        
    Returns:
    --------
    pd.DataFrame : Filtered data
    """
    filtered_data = data.copy()
    
    for col, (min_val, max_val) in thresholds.items():
        if col in filtered_data.columns:
            if min_val is not None:
                filtered_data = filtered_data[filtered_data[col] >= min_val]
            if max_val is not None:
                filtered_data = filtered_data[filtered_data[col] <= max_val]
    
    return filtered_data

