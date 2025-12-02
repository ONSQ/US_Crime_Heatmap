import pandas as pd
import streamlit as st
import numpy as np
import config
from typing import Tuple, Optional
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

# Initialize geocoder
geolocator = Nominatim(user_agent="us_crime_heatmap")

# --- DATA LOADING ---
@st.cache_data
def load_data(year=None):
    """
    Loads the crime data for a specific year.
    If no year is specified, loads the default year.
    """
    if year is None:
        year = config.DEFAULT_YEAR
        
    if year not in config.DATA_FILES:
        st.error(f"Data for year {year} not found.")
        return pd.DataFrame()

    filename = config.DATA_FILES[year]
    try:
        df = pd.read_csv(filename)
        # Clean column names - replace dots with spaces for readability
        df.columns = df.columns.str.strip().str.replace("\\n", " ")
        
        # Rename key columns for easier access
        if "County.2" in df.columns:
            df["County_Name"] = df["County.2"]
        elif "County" in df.columns:
            df["County_Name"] = df["County"]
            
        # Handle population column
        if "Percent..SEX.AND.AGE..Total.population" in df.columns:
            df["Population"] = pd.to_numeric(df["Percent..SEX.AND.AGE..Total.population"], errors="coerce")
        
        # Clean crime column names - replace dots with spaces
        crime_cols_mapping = {
            "Violent.crime": "Violent Crime",
            "Murder.and.nonnegligent.manslaughter": "Murder",
            "Rape": "Rape",
            "Robbery": "Robbery",
            "Aggravated.assault": "Aggravated Assault",
            "Property.crime": "Property Crime",
            "Burglary": "Burglary",
            "Larceny..theft": "Larceny-Theft",
            "Motor.vehicle.theft": "Motor Vehicle Theft",
            "Arson1": "Arson"
        }
        
        for old_name, new_name in crime_cols_mapping.items():
            if old_name in df.columns:
                df[new_name] = pd.to_numeric(df[old_name], errors="coerce")
        
        # Calculate Total Offenses
        if "Violent Crime" in df.columns and "Property Crime" in df.columns:
            df["Total Offenses"] = df["Violent Crime"].fillna(0) + df["Property Crime"].fillna(0)
        
        return df
    except FileNotFoundError:
        st.error(f"File not found: {filename}")
        return pd.DataFrame()

@st.cache_data
def load_all_years():
    """
    Loads data from all configured years and combines them into a single DataFrame.
    Adds a 'Year' column.
    """
    all_data = []
    for year, filename in config.DATA_FILES.items():
        try:
            df = load_data(year)
            if not df.empty:
                df['Year'] = year
                all_data.append(df)
        except Exception:
            continue
            
    if not all_data:
        return pd.DataFrame()
        
    return pd.concat(all_data, ignore_index=True)

# --- GEOCODING ---
@st.cache_data
def load_county_coordinates():
    """Load pre-computed county coordinates from CSV file."""
    try:
        coords_df = pd.read_csv("county_coordinates.csv")
        return coords_df
    except FileNotFoundError:
        st.warning("County coordinates file not found. Using fallback geocoding (slower).")
        return pd.DataFrame()

def add_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds latitude and longitude columns to the dataframe.
    Uses pre-computed coordinates for fast loading.
    """
    if "County_Name" not in df.columns or "State" not in df.columns:
        df["latitude"] = None
        df["longitude"] = None
        return df
    
    # Load pre-computed coordinates
    coords_df = load_county_coordinates()
    
    if not coords_df.empty:
        # Merge with pre-computed coordinates
        df = df.merge(
            coords_df[["State", "County_Name", "latitude", "longitude"]], 
            on=["State", "County_Name"], 
            how="left"
        )
    else:
        # Fallback: no coordinates available
        df["latitude"] = None
        df["longitude"] = None
    
    return df

# --- DATA PROCESSING ---
@st.cache_data
def merge_and_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processes county data and calculates crime percentages."""
    
    # Calculate crime percentage columns
    for col in df.columns:
        if col not in config.EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[col]):
            if "Population" in df.columns and col != "Population":
                # Avoid division by zero
                df[f"{col} %"] = df.apply(
                    lambda row: (row[col] / row["Population"]) * 100 if row["Population"] > 0 else 0,
                    axis=1
                )
                
    return df

# --- PREDICTION ---
def predict_crime_growth(df: pd.DataFrame, crime_col: str, current_pop: int, growth_rate: float = config.DEFAULT_GROWTH_RATE) -> Tuple[Optional[int], Optional[int]]:
    """
    Predicts crime range based on population growth using linear regression.
    Returns (lower_bound, upper_bound).
    """
    if crime_col in df.columns and "Population" in df.columns:
        valid_data = df.dropna(subset=[crime_col, "Population"])
        if len(valid_data) > 1:
            x = valid_data["Population"].values
            y = valid_data[crime_col].values
            
            try:
                slope, intercept = np.polyfit(x, y, 1)
                
                future_pop_lower = current_pop * (1 - growth_rate)
                future_pop_upper = current_pop * (1 + growth_rate)
                
                pred_crime_lower = slope * future_pop_lower + intercept
                pred_crime_upper = slope * future_pop_upper + intercept
                
                return int(max(0, pred_crime_lower)), int(max(0, pred_crime_upper))
            except Exception:
                return None, None
    return None, None
