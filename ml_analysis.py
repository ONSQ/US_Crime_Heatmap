"""
Machine Learning and Statistical Analysis Functions
"""
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, List
import plotly.express as px
import plotly.graph_objects as go

# --- CLUSTERING ---
@st.cache_data
def perform_clustering(df: pd.DataFrame, n_clusters: int = 5) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Perform K-Means clustering on crime and demographic data.
    Returns: (df with cluster labels, cluster centers, feature names)
    """
    # Select features for clustering
    feature_cols = [
        'Violent Crime', 'Property Crime', 'Murder', 'Robbery',
        'Population', 'Median_Income_2023'
    ]
    
    # Add demographic features if available
    demographic_cols = [col for col in df.columns if 'Percent' in col and 'SEX.AND.AGE' in col]
    if demographic_cols:
        feature_cols.extend(demographic_cols[:5])  # Add first 5 demographic features
    
    # Filter to available columns
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Prepare data
    X = df[available_features].copy()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    
    # --- DYNAMIC CLUSTER NAMING ---
    # Calculate means for each cluster
    cluster_means = df_clustered.groupby('Cluster')[['Violent Crime', 'Property Crime', 'Population', 'Median_Income_2023']].mean()
    
    # Calculate global means
    global_means = df[['Violent Crime', 'Property Crime', 'Population', 'Median_Income_2023']].mean()
    
    cluster_names = {}
    for cluster_id in range(n_clusters):
        means = cluster_means.loc[cluster_id]
        
        # Determine characteristics relative to global average
        # Population
        if means['Population'] > global_means['Population'] * 2.5:
            pop_tag = "Major Metro"
        elif means['Population'] > global_means['Population'] * 1.2:
            pop_tag = "Urban"
        elif means['Population'] < global_means['Population'] * 0.6:
            pop_tag = "Rural"
        else:
            pop_tag = "Suburban/Mixed"
            
        # Income
        if means['Median_Income_2023'] > global_means['Median_Income_2023'] * 1.15:
            inc_tag = "Wealthy"
        elif means['Median_Income_2023'] < global_means['Median_Income_2023'] * 0.85:
            inc_tag = "Low Income"
        else:
            inc_tag = "Middle Income"
            
        # Crime (Total)
        total_crime = means['Violent Crime'] + means['Property Crime']
        global_total = global_means['Violent Crime'] + global_means['Property Crime']
        
        if total_crime > global_total * 1.5:
            crime_tag = "High Crime"
        elif total_crime < global_total * 0.6:
            crime_tag = "Safe"
        else:
            crime_tag = "Avg Crime"
            
        # Construct Name
        # Prioritize distinctive features to keep names short
        name_parts = []
        
        if pop_tag == "Major Metro":
            name_parts.append("Metro")
        elif pop_tag == "Rural":
            name_parts.append("Rural")
            
        if inc_tag == "Wealthy":
            name_parts.append("Wealthy")
        elif inc_tag == "Low Income":
            name_parts.append("Low Income")
            
        if crime_tag == "High Crime":
            name_parts.append("High Crime")
        elif crime_tag == "Safe":
            name_parts.append("Safe")
            
        # Fallback if generic
        if not name_parts:
            name_parts.append("Average County")
            
        cluster_names[cluster_id] = ", ".join(name_parts)

    df_clustered['Cluster_Label'] = df_clustered['Cluster'].map(cluster_names)
    
    return df_clustered, kmeans.cluster_centers_, available_features

# --- FEATURE IMPORTANCE ---
@st.cache_data
def calculate_feature_importance(df: pd.DataFrame, target_col: str = 'Total Offenses') -> pd.DataFrame:
    """
    Calculate feature importance using Random Forest.
    Returns: DataFrame with features and their importance scores.
    """
    # Select features
    exclude_cols = {
        'State', 'County', 'County.2', 'County_Name', 'latitude', 'longitude',
        'Cluster', 'Cluster_Label', target_col, 'Metropolitan.Nonmetropolitan'
    }
    
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols 
                   and pd.api.types.is_numeric_dtype(df[col])
                   and not col.endswith('%')]
    
    # Prepare data
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Remove rows with missing target
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_imputed, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return importance_df

# --- OUTLIER DETECTION ---
@st.cache_data
def detect_outliers(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Detect outlier counties using Isolation Forest.
    Returns: DataFrame with outlier labels.
    """
    # Select features
    feature_cols = ['Violent Crime', 'Property Crime', 'Population', 'Median_Income_2023']
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Prepare data
    X = df[available_features].copy()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Detect outliers
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(X_imputed)
    
    # Add outlier labels
    df_outliers = df.copy()
    df_outliers['Is_Outlier'] = outliers == -1
    df_outliers['Outlier_Label'] = df_outliers['Is_Outlier'].apply(
        lambda x: 'Outlier' if x else 'Normal'
    )
    
    return df_outliers

# --- VISUALIZATION HELPERS ---
def create_cluster_map(df: pd.DataFrame, crime_col: str = 'Total Offenses'):
    """Create an interactive Plotly map colored by clusters."""
    df_plot = df.dropna(subset=['latitude', 'longitude', 'Cluster'])
    
    fig = px.scatter_geo(
        df_plot,
        lat='latitude',
        lon='longitude',
        color='Cluster_Label',
        hover_name='County_Name',
        hover_data={
            'State': True,
            'Population': ':,',
            crime_col: ':,',
            'Cluster_Label': True,
            'latitude': False,
            'longitude': False
        },
        scope='usa',
        title=f'County Clusters Based on Crime Patterns',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(height=600)
    return fig

def create_feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 15):
    """Create a horizontal bar chart of feature importance."""
    top_features = importance_df.head(top_n)
    
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top {top_n} Most Important Features for Predicting Crime',
        labels={'Importance': 'Feature Importance', 'Feature': ''},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    return fig

def create_income_crime_scatter(df: pd.DataFrame, crime_col: str = 'Total Offenses'):
    """Create scatter plot of income vs crime with regression line."""
    df_plot = df.dropna(subset=['Median_Income_2023', crime_col, 'Population'])
    
    # Calculate crime rate per capita
    df_plot['Crime_Rate'] = (df_plot[crime_col] / df_plot['Population']) * 100
    
    fig = px.scatter(
        df_plot,
        x='Median_Income_2023',
        y='Crime_Rate',
        hover_name='County_Name',
        hover_data={
            'State': True,
            'Population': ':,',
            crime_col: ':,',
            'Median_Income_2023': ':$,',
            'Crime_Rate': ':.2f'
        },
        trendline='ols',
        title=f'Median Income vs Crime Rate',
        labels={
            'Median_Income_2023': 'Median Income ($)',
            'Crime_Rate': f'{crime_col} Rate (%)'
        },
        color='Population',
        size='Population',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=500)
    return fig

def create_outlier_map(df: pd.DataFrame):
    """Create a map highlighting outlier counties."""
    df_plot = df.dropna(subset=['latitude', 'longitude', 'Is_Outlier'])
    
    fig = px.scatter_geo(
        df_plot,
        lat='latitude',
        lon='longitude',
        color='Outlier_Label',
        hover_name='County_Name',
        hover_data={
            'State': True,
            'Population': ':,',
            'Total Offenses': ':,',
            'Outlier_Label': True,
            'latitude': False,
            'longitude': False
        },
        scope='usa',
        title='Outlier Counties (Unusual Crime Patterns)',
        color_discrete_map={'Normal': 'lightblue', 'Outlier': 'red'}
    )
    
    fig.update_layout(height=600)
    return fig
