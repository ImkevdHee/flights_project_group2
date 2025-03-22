# flights_project_group2

## Project Overview
This project focuses on the analysis and visualization of flight and airport data based on the NYC flights dataset for the year 2023. The project follows a structured, multi-part assignment approach as required by the course, resulting in geospatial visualizations, clustering analyses, database queries, data wrangling, and an interactive dashboard.

## Project Structure
```
flights_project_group2/
├─ part1.py        # Visual exploration of airport data
├─ part3.py        # SQL-based analysis and queries on flights data
├─ part4.py        # Data wrangling, consistency checks, and feature engineering
├─ part5.py        # Streamlit dashboard for visualization and interaction
├─ flights_database.db # SQLite database with all flights, airports, weather, planes, and airlines data
├─ airports.csv    # Metadata on airports worldwide
├─ requirements.txt # List of dependencies for running all scripts and the dashboard
└─ README.md       # Project documentation
```

## Features
- Interactive visualizations using Plotly and Matplotlib
- K-Means clustering of airports
- Euclidean and Geodesic distance calculations
- SQL queries for route analysis, airline performance, and delay distributions
- Analysis of relationships between weather patterns and flight delays
- Network analysis of major flight routes
- Fully interactive dashboard built with Streamlit
- Dynamic filtering by departure and arrival airports, dates, and time periods

## Installation Instructions
1. Clone this repository:
```
git clone https://github.com/[your-github-username]/[repository-name].git
cd [repository-name]
```
2. (Optional) Create and activate a virtual environment:
```
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
```
3. Install all required dependencies:
```
pip install -r requirements.txt
```

## Running the Scripts
### Part 1 — Airport Data Visualizations
```
python part1.py
```
This will generate geospatial maps and clustering plots.

### Part 3 — SQL Queries and Flight Data Analysis
```
python part3.py
```
Executes SQL queries on the provided SQLite database and outputs relevant statistics and visualizations.

### Part 4 — Data Wrangling and Consistency Checks
```
python part4.py
```
Performs checks for missing values, data consistency, and adds engineered features.

### Part 5 — Running the Interactive Dashboard
1. Ensure `part5.py` is present in your working directory.
2. Download and extract `flights_processed.zip` in the same folder.
3. Launch the dashboard with the following command:
```
streamlit run part5.py
```
The dashboard will open in your default web browser.

## Dashboard Functionalities
- Display of general statistics (total flights, average delays, most frequent destinations)
- Histogram and pie chart visualizations for arrival delays
- Date-based statistics and filtering
- Interactive analysis of delays by hour of the day
- Weather and delay relationship visualizations
- Detailed statistics per route, including average speeds and airline performance
- General insights tab with highlights of busiest routes and delay-prone aircraft models

## Database Overview
- **airlines**: Contains airline codes and names
- **airports**: Includes airport location, altitude, and timezone information
- **flights**: Comprehensive data on over 425,000 flights with delays, distance, and schedules
- **planes**: Aircraft details with tail numbers, manufacturers, and models
- **weather**: Hourly weather conditions at NYC airports

## Deployment
To deploy the dashboard on Streamlit Cloud:
1. Ensure all project files are pushed to GitHub.
2. Visit https://streamlit.io/cloud.
3. Click on 'New app' and connect your GitHub repository.
4. Select the correct branch and point to `part5.py`.
5. Click 'Deploy' to make the dashboard publicly accessible.

## Requirements
The `requirements.txt` file includes:
```
streamlit
pandas
numpy
plotly
sqlite3
geopy
scipy
scikit-learn
matplotlib
```

## Team Contribution
All team members contributed equally to different project parts. Contributions included database querying, data cleaning, visualization building, and dashboard deployment. The GitHub commit history reflects collaborative development and code reviews.

## Future Improvements
- Implementation of advanced filters by airline and aircraft model
- Predictive modeling for estimating future delays
- Integration of real-time weather data for live analysis
- Additional geospatial network visualizations for route optimization

## Contact Information
For questions or inquiries, please contact:
- Karam-Eddine Âbbadi
- Achraf Hmydou
- Kyboo Shubin
- Imke van der Hee

This project was developed for the Data Engineering course at VU Amsterdam, academic year 2023-2024.

