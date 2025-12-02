# Configuration settings for the US County Crime Heatmap App

# Data Files
DATA_FILES = {
    2023: "crime_data_county.csv",
    # Add future years here, e.g.:
    # 2022: "crime_data_county_2022.csv",
}
# Default to the latest year
DEFAULT_YEAR = max(DATA_FILES.keys())

# Map Settings - US Center
DEFAULT_LAT = 39.8  # Center of continental US
DEFAULT_LON = -98.6  # Center of continental US
DEFAULT_ZOOM = 4  # Show entire US
COUNTY_ZOOM = 7  # Zoom level for individual county

# Prediction Settings
DEFAULT_GROWTH_RATE = 0.05

# Columns to Exclude from Crime Analysis
EXCLUDE_COLS = {
    "State", "Metropolitan.Nonmetropolitan", "County", "County.2",
    "latitude", "longitude", "Population", "Median_Income_2023",
    "Percent..Total.housing.units"
}
