# part1.py

import pandas as pd
import plotly.express as px
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("airports.csv")

print(df.head())

fig = px.scatter_geo(df,
                     lat='lat',
                     lon='lon',
                     hover_name='name',
                     color = 'alt',
                     title="Airports around the World",
                     projection='natural earth')

fig.show()


df_clean = df.dropna(subset=['tzone'])
us_airports = df_clean[df_clean['tzone'].str.startswith('America/')]

fig_us = px.scatter_geo(us_airports,
                        lat='lat',
                        lon='lon',
                        hover_name='name',
                        color = 'alt',
                        title="Airports in the United States",
                        projection='albers usa')
fig_us.show()


nyc_lat = 40.7128
nyc_lon = -74.0060

def plot_airports_lines(faa_codes):
    fig = go.Figure()

    for faa_code in faa_codes:
        airport = df[df['faa'] == faa_code]
        
        if airport.empty:
            print(f"Airport with FAA code '{faa_code}' not found.")
            continue
        
        airport_name = airport['name'].values[0]
        airport_lat = airport['lat'].values[0]
        airport_lon = airport['lon'].values[0]
        
        if not airport['tzone'].str.startswith('America/').any():
            print(f"{airport_name} is located outside the US.")
            continue
        
        projection = 'albers usa'
        fig.add_trace(go.Scattergeo(
            locationmode='USA-states',
            lon=[nyc_lon, airport_lon],
            lat=[nyc_lat, airport_lat],
            mode='lines+markers',
            marker=dict(color='red', size=8),
            line=dict(width=2, color='blue'),
            text=[f"NYC", airport_name],
            hoverinfo='text+lat+lon'
        ))

    fig.update_layout(
        title=f"NYC to Multiple Airports",
        geo=dict(
            projection_type=projection,
            showland=True,
            landcolor='white',
            countrycolor='lightgray'
        )
    )

    fig.show()

def user_input():
    faa_codes_input = input("Enter a comma-separated list of FAA codes of airports (e.g., LAX, JFK, BOS): ")
    faa_codes = [code.strip() for code in faa_codes_input.split(",")]
    plot_airports_lines(faa_codes)

user_input()


JFK_lat = 40.641766
JFK_lon = -73.780968

def euclidean_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

df["distance_from_JFK"] = df.apply(lambda row: euclidean_distance(JFK_lat, JFK_lon, row["lat"], row["lon"]), axis=1)

plt.figure(figsize=(8, 5))
plt.hist(df["distance_from_JFK"], bins=30, edgecolor='black')
plt.title('Distribution of Euclidean Distances from JFK Airport')
plt.xlabel('Distance (degrees)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Earth's radius in kilometers
R = 6.378

def geodesic_distance(lat1, lon1, lat2, lon2, R):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    
    phi_m = (lat1 + lat2) / 2.0
    
    delta_phi = lat2 - lat1
    delta_lambda = lon2 - lon1
    
    term1 = ((2*np.sin(delta_phi / 2)) * np.cos(delta_lambda / 2))**2
    term2 = (2*np.cos(phi_m) * np.sin(delta_lambda / 2))**2
    
    return R * np.sqrt(term1 + term2)

df["geodesic_distance_from_JFK"] = df.apply(
    lambda row: geodesic_distance(JFK_lat, JFK_lon, row["lat"], row["lon"], R), axis=1
)

plt.figure(figsize=(8, 5))
plt.hist(df["geodesic_distance_from_JFK"], bins=30, edgecolor='black')
plt.title('Distribution of Geodesic Distances from JFK Airport')
plt.xlabel('Distance (km)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



tz_counts = df["tz"].value_counts().sort_index()

plt.figure(figsize=(10, 5))
tz_counts.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Distribution of Airports Across Time Zones")
plt.xlabel("Time Zone (UTC Offset)")
plt.ylabel("Number of Airports")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()



# Extra analysis voor creative points

from sklearn.cluster import KMeans

num_clusters = 5
coordinates = df[['lat', 'lon']]
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(coordinates)

fig = px.scatter_geo(df, lat='lat', lon='lon', hover_name='name', color=df['cluster'].astype(str),
                     title="Airport Clusters", projection='natural earth')
fig.show()
