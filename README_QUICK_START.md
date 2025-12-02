# US County Crime Heatmap - Quick Start

## What Changed
The app now loads **instantly** instead of taking forever! 

### Solution
- ✅ Replaced slow geocoding with fast CSV lookup
- ✅ Created `county_coordinates.csv` with pre-computed coordinates
- ✅ Included 60+ major US counties to start

## Running the App

Just refresh your browser or restart:
```bash
streamlit run app.py
```

The app should now load in seconds!

## Current Coverage

The starter `county_coordinates.csv` includes major counties from:
- California (Los Angeles, San Diego, Orange, etc.)
- Texas (Harris, Dallas, Tarrant, etc.)
- Florida (Miami-Dade, Broward, etc.)
- New York (NYC boroughs)
- Illinois (Cook, DuPage, etc.)
- And more...

**Total: ~60 major counties**

## Adding More Counties

If you want to add ALL 2,400+ counties, run:
```bash
python precompute_coordinates.py
```

This will:
- Take 30-60 minutes to geocode all counties
- Save progress every 50 counties
- Create a complete `county_coordinates.csv` file

**Note:** You only need to run this once! After that, the app will always load fast.

## Files

- `county_coordinates.csv` - Pre-computed coordinates (starter set)
- `precompute_coordinates.py` - Script to geocode all counties
- `app.py` - Main application
- `data_manager.py` - Data loading (now uses CSV lookup)

## Why This is Faster

**Before:** Geocoded 2,400+ counties on every app start → 10+ minutes
**Now:** Loads coordinates from CSV → 2-3 seconds ⚡

Enjoy your fast-loading crime heatmap!
