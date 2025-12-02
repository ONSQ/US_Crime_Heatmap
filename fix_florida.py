import pandas as pd
import os

files_to_check = [
    "crime_data_county.csv",
    "county_coordinates.csv",
    "county_coordinates_temp.csv"
]

for filename in files_to_check:
    if os.path.exists(filename):
        print(f"Checking {filename}...")
        try:
            # Read file
            df = pd.read_csv(filename)
            
            # Check if 'State' column exists
            if "State" in df.columns:
                # Check for FLORIDA2
                mask = df["State"] == "FLORIDA2"
                count = mask.sum()
                
                if count > 0:
                    print(f"Found {count} instances of 'FLORIDA2' in {filename}. Fixing...")
                    # Replace
                    df.loc[mask, "State"] = "FLORIDA"
                    # Save back
                    df.to_csv(filename, index=False)
                    print(f"Fixed {filename}.")
                else:
                    print(f"No 'FLORIDA2' found in {filename}.")
            else:
                print(f"Column 'State' not found in {filename}.")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    else:
        print(f"{filename} not found.")
