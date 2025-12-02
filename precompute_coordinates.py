"""
Pre-compute county coordinates and save to CSV for faster loading.
Run this once to generate county_coordinates.csv
"""
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

# Initialize geocoder
geolocator = Nominatim(user_agent="us_crime_heatmap_precompute")

def geocode_county(county_name, state, max_retries=3):
    """Geocode a county to get its latitude and longitude."""
    clean_county = county_name.replace(" County", "").strip()
    
    queries = [
        f"{clean_county} County, {state}, USA",
        f"{clean_county}, {state}, USA",
    ]
    
    for query in queries:
        for attempt in range(max_retries):
            try:
                location = geolocator.geocode(query, timeout=10)
                if location:
                    return location.latitude, location.longitude
                time.sleep(0.5)
            except (GeocoderTimedOut, GeocoderServiceError):
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    break
    
    return None, None

# Load county data
print("Loading county data...")
df = pd.read_csv("crime_data_county.csv")

# Get unique counties
if "County.2" in df.columns:
    df["County_Name"] = df["County.2"]
elif "County" in df.columns:
    df["County_Name"] = df["County"]

unique_counties = df[["State", "County_Name"]].drop_duplicates()
print(f"Found {len(unique_counties)} unique counties to geocode")

# Check for existing progress
results = []
try:
    existing_df = pd.read_csv("county_coordinates_temp.csv")
    
    # Only consider valid coordinates as "processed"
    # This forces a retry for any NaNs (like the previous FLORIDA2 failures)
    valid_df = existing_df.dropna(subset=['latitude', 'longitude'])
    results = valid_df.to_dict('records')
    print(f"Loaded {len(results)} valid geocoded counties from temp file.")
    
    # Create a set of processed (State, County) tuples for fast lookup
    processed_counties = set((row['State'], row['County_Name']) for row in results)
except FileNotFoundError:
    print("No temp file found, starting from scratch.")
    processed_counties = set()

# Geocode each county
total_counties = len(unique_counties)
for idx, row in unique_counties.iterrows():
    state = row["State"]
    county = row["County_Name"]
    
    # Skip invalid rows
    if pd.isna(state) or pd.isna(county):
        continue
        
    # Skip already processed
    if (state, county) in processed_counties:
        continue
    
    print(f"Geocoding {len(results)+1}/{total_counties}: {county}, {state}...")
    lat, lon = geocode_county(county, state)
    
    results.append({
        "State": state,
        "County_Name": county,
        "latitude": lat,
        "longitude": lon
    })
    
    # Save progress every 20 counties
    if len(results) % 20 == 0:
        temp_df = pd.DataFrame(results)
        temp_df.to_csv("county_coordinates_temp.csv", index=False)
        print(f"Progress saved: {len(results)}/{total_counties}")

# Save final results
coords_df = pd.DataFrame(results)
coords_df.to_csv("county_coordinates.csv", index=False)
print(f"\nDone! Saved coordinates for {len(coords_df)} counties to county_coordinates.csv")
print(f"Successfully geocoded: {coords_df['latitude'].notna().sum()} counties")
print(f"Failed to geocode: {coords_df['latitude'].isna().sum()} counties")
