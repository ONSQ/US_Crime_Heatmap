import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from folium import CircleMarker, Tooltip
from streamlit_folium import st_folium

import config
import data_manager
import ml_analysis

# --- APP CONFIG ---
st.set_page_config(layout="wide")
st.title("US County Crime Rate, Age and Race Statistics Interactive Heatmap")
st.caption("The 48 continental states are included in this analysis.")
st.caption("Search US counties and visualize crime by offense type.")
st.caption("Search US counties and visualize Race and Age Statistics.")
st.caption("Data from the 2024 FBI Crime Data and 2020 US Census. All crimes listed are preadjudicated and not convictions.")    
st.caption("Webapp by Owen Eskew (WGN372), Data Crafting by Torben Rehnert (ODS799), Data analysis and presentation by Emmanuel Amoah (DSN270)")


# --- DATA LOADING ---
st.sidebar.header("📂 Data Management")

# Year Selector
available_years = sorted(list(config.DATA_FILES.keys()), reverse=True)
selected_year = st.sidebar.selectbox("Select Year", available_years, index=0)

uploaded_file = st.sidebar.file_uploader("Upload your own Crime CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.replace("\n", " ")
        st.sidebar.success("Custom data loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        st.stop()
else:
    df = data_manager.load_data(selected_year)

# Ensure required columns exist
if "County_Name" not in df.columns:
    st.error("Column 'County_Name' not found in data")
    st.stop()

if "State" not in df.columns:
    st.error("Column 'State' not found in data")
    st.stop()

# Process data
df = data_manager.merge_and_process_data(df)

# Add coordinates (fast CSV lookup)
df = data_manager.add_coordinates(df)

# --- SIDEBAR ---
# Search bar
search_county = st.sidebar.text_input("Search for a US county:")

# State filter
if "State" in df.columns:
    states = ["All States"] + sorted(df["State"].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("Filter by State:", states)
    if selected_state != "All States":
        df = df[df["State"] == selected_state]

# Identify all numeric crime-related columns (absolute numbers only)
crime_types = [col for col in df.columns if col not in config.EXCLUDE_COLS 
               and pd.api.types.is_numeric_dtype(df[col]) and "%" not in col]

if crime_types:
    # Sort the list and move "Total Offenses" to the top if it exists
    sorted_crime_types = sorted(crime_types)
    if "Total Offenses" in sorted_crime_types:
        sorted_crime_types.remove("Total Offenses")
        sorted_crime_types.insert(0, "Total Offenses")

    # Dropdown default = first item (Total Offenses)
    crime_col = st.sidebar.selectbox(
        "Choose crime category (absolute numbers):",
        sorted_crime_types,
        index=0
    )
else:
    st.error("No numeric crime category columns found in the dataset.")
    st.stop()

# Toggle for using % vs absolute
use_percentage = st.sidebar.checkbox("Use % (crime per population) for heatmap", value=False)

# --- PREPARE DATA FOR DISPLAY ---
# Ensure numeric
df[crime_col] = pd.to_numeric(df[crime_col], errors="coerce")
df_heat = df.dropna(subset=["latitude", "longitude", crime_col])

# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Heatmap & Search", "📊 Data Analysis", "🔮 Prediction Model", "🧠 Statistical Analysis"])

with tab1:
    # --- METRICS ROW ---
    # Show statewide summary info at the top
    total_population = df["Population"].sum()
    total_crime = df[crime_col].sum()
    
    st.markdown("### US County Crime Snapshot")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total Population", f"{int(total_population):,}")
    col_m2.metric(f"Total {crime_col}", f"{int(total_crime):,}")
    
    if total_population > 0:
        percent_value = (total_crime / total_population) * 100
        col_m3.metric("Crime Rate", f"{percent_value:.2f}%")

    # --- MAP SECTION ---
    st.subheader(f"{crime_col} Heatmap")

    avg_lat, avg_lon = config.DEFAULT_LAT, config.DEFAULT_LON
    zoom_level = config.DEFAULT_ZOOM

    if search_county:
        results = df[df["County_Name"].str.lower().str.contains(search_county.lower(), na=False)]
        if not results.empty:
            county = results.iloc[0]
            if pd.notnull(county["latitude"]) and pd.notnull(county["longitude"]):
                avg_lat, avg_lon = county["latitude"], county["longitude"]
                zoom_level = config.COUNTY_ZOOM
            
            # Display county stats in an expander or just below
            with st.expander(f"Details for {county['County_Name']}, {county['State']}", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Coordinates:** ({county['latitude']:.4f}, {county['longitude']:.4f})")
                    if pd.notnull(county['Population']):
                        st.write(f"**Population:** {int(county['Population']):,}")
                    else:
                        st.write("**Population:** N/A")
                with c2:
                    abs_val = county[crime_col]
                    pct_val = county.get(f"{crime_col} %")
                    if pd.notnull(abs_val):
                        st.write(f"**{crime_col}:** {abs_val:,.0f}")
                    if pd.notnull(pct_val):
                        st.write(f"**Rate:** {pct_val:.2f}%")
        else:
            st.warning("County not found. Try a different spelling?")

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=zoom_level)

    # Decide heatmap data
    if use_percentage:
        percent_col = f"{crime_col} %"
        if percent_col not in df_heat.columns:
            st.error(f"No percentage data found for {crime_col}.")
            st.stop()

        heat_data = [
            [row["latitude"], row["longitude"], row[percent_col]]
            for _, row in df_heat.iterrows()
            if pd.notnull(row[percent_col]) and row[percent_col] > 0
        ]
    else:
        heat_data = [
            [row["latitude"], row["longitude"], row[crime_col]]
            for _, row in df_heat.iterrows()
            if pd.notnull(row[crime_col]) and row[crime_col] > 0
        ]

    if heat_data:
        HeatMap(
            heat_data,
            radius=15,
            blur=10,
            max_zoom=9,
            gradient={0.0: 'blue', 0.2: 'green', 0.4: 'yellow', 0.6: 'orange', 0.8: 'red', 1.0: 'darkred'}
        ).add_to(m)

    # Add invisible markers with improved tooltips
    for _, row in df_heat.iterrows():
        abs_value = row[crime_col]
        percent_value = row.get(f"{crime_col} %", None)

        # HTML styled tooltip
        pop_display = f"{int(row['Population']):,}" if pd.notnull(row['Population']) else "N/A"
        crime_display = f"{abs_value:,.0f}" if pd.notnull(abs_value) else "N/A"
        
        tooltip_text = f"""
        <div style="font-family: sans-serif; font-size: 12px;">
            <b>{row['County_Name']}, {row['State']}</b><br>
            Population: {pop_display}<br>
            {crime_col}: {crime_display}<br>
        """
        if percent_value is not None and pd.notnull(percent_value):
            tooltip_text += f"Rate: {percent_value:.2f}%"
        tooltip_text += "</div>"

        CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color="#00000000",
            fill=True,
            fill_color="#000000",
            fill_opacity=0.01, # Almost invisible
            tooltip=Tooltip(tooltip_text, sticky=True)
        ).add_to(m)

    st_folium(m, width=900, height=600)

with tab2:
    # --- TOP/BOTTOM TABLES ---
    st.subheader(f"Safest & Most Dangerous Counties by {crime_col}")
    
    # Add a filter for population to make this more meaningful
    min_pop = st.slider("Minimum Population Filter", 0, 1000000, 0, 1000)
    
    filtered_df = df_heat[df_heat["Population"] >= min_pop]
    
    percent_col = f"{crime_col} %"
    cols_to_show = ["County_Name", "State", "Population", crime_col]
    if percent_col in filtered_df.columns:
        cols_to_show.append(percent_col)

    sorted_df = filtered_df[cols_to_show].sort_values(by=percent_col if percent_col in filtered_df.columns else crime_col)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ✅ Safest (Lowest Crime)")
        st.dataframe(sorted_df.head(10).reset_index(drop=True), use_container_width=True)
    with col2:
        st.markdown("#### ⚠️ Most Dangerous (Highest Crime)")
        st.dataframe(sorted_df.tail(10).sort_values(by=percent_col if percent_col in filtered_df.columns else crime_col, ascending=False).reset_index(drop=True), use_container_width=True)

    # Download Button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name='us_county_crime_filtered.csv',
        mime='text/csv',
    )

    st.divider()
    
    # --- CORRELATION ANALYSIS ---
    st.subheader("🔗 Correlation Analysis")
    st.caption("How different variables relate to each other (1.0 = perfect positive correlation, -1.0 = perfect negative correlation).")
    
    # Select numeric columns only
    numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
    # Drop columns that are not useful for correlation
    cols_to_drop = ["latitude", "longitude"]
    numeric_df = numeric_df.drop(columns=[c for c in cols_to_drop if c in numeric_df.columns], errors='ignore')
    
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm", axis=None), use_container_width=True)
    else:
        st.info("Not enough numeric data for correlation analysis.")

with tab3:
    # --- PREDICTION ---
    st.subheader("🔮 Future Trend Prediction")
    st.info("This model uses linear regression on the current dataset to estimate crime levels based on population growth.")

    # Determine population value
    if search_county:
        results = df[df["County_Name"].str.lower().str.contains(search_county.lower(), na=False)]
        if not results.empty:
            county = results.iloc[0]
            if pd.notnull(county["Population"]):
                pop_int = int(county["Population"])
            else:
                pop_int = int(df["Population"].sum())
            st.markdown(f"### Prediction for **{county['County_Name']}, {county['State']}**")
        else:
            pop_int = int(df["Population"].sum())
            st.markdown("### National Prediction")
    else:
        pop_int = int(df["Population"].sum())
        st.markdown("### National Prediction")

    lower, upper = data_manager.predict_crime_growth(df, crime_col, pop_int)

    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        if pop_int > 0:
            st.write(f"**Current Population:** {pop_int:,}")
        else:
            st.write("**Current Population:** N/A")
        st.write(
            f"**Predicted Population (±5%):** "
            f"{int(pop_int * 0.95):,} - {int(pop_int * 1.05):,}"
        )

    with col_p2:
        if lower is not None and upper is not None:
            lower_percent = (lower / pop_int) * 100
            upper_percent = (upper / pop_int) * 100
            
            st.metric(
                label=f"Estimated {crime_col} Range",
                value=f"{lower:,} - {upper:,}",
                delta=None
            )
            st.caption(f"Implied Rate: {lower_percent:.2f}% - {upper_percent:.2f}%")
        else:
            st.warning("Could not calculate prediction based on current data.")

    # --- VISUALIZATION ---
    st.markdown("#### 📈 Visualizing the Trend")
    
    # Prepare data for scatter plot
    if crime_col in df.columns and "Population" in df.columns:
        chart_data = df.dropna(subset=[crime_col, "Population"])
        
        # Add the prediction point if available
        if lower is not None and upper is not None:
            # We plot the average prediction
            pred_val = (lower + upper) / 2
            # Create a small dataframe for the prediction point
            pred_df = pd.DataFrame({
                "Population": [int(pop_int * 1.05)], # Future pop
                crime_col: [pred_val],
                "Type": ["Prediction"]
            })
            chart_data["Type"] = "Historical"
            combined_data = pd.concat([chart_data, pred_df], ignore_index=True)
            
            st.scatter_chart(
                combined_data,
                x="Population",
                y=crime_col,
                color="Type",
                size=20
            )
        else:
            st.scatter_chart(chart_data, x="Population", y=crime_col)

# --- TREND ANALYSIS (If multiple years exist) ---
if len(config.DATA_FILES) > 1:
    with st.expander("📈 Trend Analysis (Multi-Year)"):
        st.subheader("Time Series Analysis")
        
        all_years_df = data_manager.load_all_years()
        
        if not all_years_df.empty:
            # County selection for trend
            counties = sorted(all_years_df["County_Name"].dropna().unique())
            selected_counties_trend = st.multiselect("Select Counties to Compare", counties, default=counties[:3] if len(counties) > 3 else counties)
            
            if selected_counties_trend:
                trend_data = all_years_df[all_years_df["County_Name"].isin(selected_counties_trend)]
                
                if crime_col in trend_data.columns:
                    st.line_chart(
                        trend_data,
                        x="Year",
                        y=crime_col,
                        color="County_Name"
                    )
                else:
                    st.warning(f"Column '{crime_col}' not found in multi-year data.")
            else:
                st.info("Select counties to see the trend.")
        else:
            st.warning("Could not load multi-year data.")

with tab4:
    st.subheader("🧠 Advanced Statistical Analysis")
    st.caption("Machine learning insights into crime patterns across US counties.")
    
    # Sub-tabs for different analyses
    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs(["Clustering", "Feature Importance", "Outlier Detection", "Income Analysis"])
    
    with ml_tab1:
        st.markdown("### 🔍 County Clustering")
        st.info("Groups counties with similar crime and demographic profiles using K-Means clustering.")
        
        n_clusters = st.slider("Number of Clusters", 2, 8, 5)
        
        if st.button("Run Clustering Analysis"):
            with st.spinner("Running K-Means Clustering..."):
                df_clustered, centers, features = ml_analysis.perform_clustering(df, n_clusters)
                
                # Show map
                fig = ml_analysis.create_cluster_map(df_clustered)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show cluster stats
                st.markdown("#### Cluster Characteristics")
                cluster_stats = df_clustered.groupby('Cluster_Label')[features].mean()
                st.dataframe(cluster_stats.style.background_gradient(cmap="Blues"), use_container_width=True)
    
    with ml_tab2:
        st.markdown("### 🌲 Feature Importance")
        st.info("Uses Random Forest to determine which factors are most predictive of crime rates.")
        
        target_col = st.selectbox("Target Variable", ["Total Offenses", "Violent Crime", "Property Crime"])
        
        if st.button("Calculate Importance"):
            with st.spinner("Training Random Forest Model..."):
                importance_df = ml_analysis.calculate_feature_importance(df, target_col)
                
                fig = ml_analysis.create_feature_importance_chart(importance_df)
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("View Detailed Importance Scores"):
                    st.dataframe(importance_df, use_container_width=True)
    
    with ml_tab3:
        st.markdown("### 🚨 Outlier Detection")
        st.info("Identifies counties with unusual crime patterns using Isolation Forest.")
        
        contamination = st.slider("Contamination (Expected % of outliers)", 0.01, 0.15, 0.05, 0.01)
        
        if st.button("Detect Outliers"):
            with st.spinner("Running Isolation Forest..."):
                df_outliers = ml_analysis.detect_outliers(df, contamination)
                
                fig = ml_analysis.create_outlier_map(df_outliers)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Detected Outliers")
                outliers = df_outliers[df_outliers['Is_Outlier']]
                st.write(f"Found {len(outliers)} outlier counties.")
                st.dataframe(
                    outliers[['County_Name', 'State', 'Population', 'Total Offenses', 'Violent Crime', 'Property Crime']], 
                    use_container_width=True
                )
    
    with ml_tab4:
        st.markdown("### 💰 Income vs Crime")
        st.info("Analyzes the relationship between median income and crime rates.")
        
        analysis_col = st.selectbox("Crime Category", ["Total Offenses", "Violent Crime", "Property Crime", "Murder", "Burglary"])
        
        fig = ml_analysis.create_income_crime_scatter(df, analysis_col)
        st.plotly_chart(fig, use_container_width=True)

