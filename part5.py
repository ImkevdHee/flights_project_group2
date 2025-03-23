import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime

conn = sqlite3.connect("flights_database.db")

flights = pd.read_sql_query("SELECT * FROM flights", conn)
weather = pd.read_sql_query("SELECT * FROM weather", conn)
planes = pd.read_sql_query("SELECT * FROM planes", conn)

st.set_page_config(layout="wide")
st.title("NYC Flights Dashboard")

# Sidebar filters
st.sidebar.header("Airport Selection")
departure_airports = st.sidebar.multiselect("Select Departure Airports", flights["origin"].unique(), default=["JFK", "LGA", "EWR"])
destinations = st.sidebar.multiselect("Select Destinations", flights["dest"].unique())

start_date = st.sidebar.date_input("Select Start Date", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Select End Date", datetime(2023, 12, 31))

part_of_day = st.sidebar.selectbox("Select Part of Day", ["Anytime", "Morning", "Afternoon", "Evening", "Night"])

flights['fl_date'] = pd.to_datetime(flights[['year', 'month', 'day']])

# Data filtering
filtered_flights = flights[
    (flights["origin"].isin(departure_airports)) &
    (flights["dest"].isin(destinations)) &
    (flights["fl_date"] >= str(start_date)) &
    (flights["fl_date"] <= str(end_date))
]

# Part of day filter
if part_of_day != "Anytime":
    hour_ranges = {"Morning": (6, 12), "Afternoon": (12, 18), "Evening": (18, 24), "Night": (0, 6)}
    h1, h2 = hour_ranges[part_of_day]
    filtered_flights = filtered_flights[(filtered_flights["dep_time"] // 100 >= h1) & (filtered_flights["dep_time"] // 100 < h2)]

# Metrics
total_flights = filtered_flights.shape[0]
avg_dep_delay = round(filtered_flights["dep_delay"].mean(), 2)
most_freq_dest = filtered_flights["dest"].mode()[0] if not filtered_flights.empty else "-"

st.metric("Total number of flights", total_flights)
st.metric("Minutes of average departure delay", avg_dep_delay)
st.metric("Most frequent destination", most_freq_dest)

# Arrival delay distribution
st.subheader("Arrival Delay Distribution")
fig_delay = px.histogram(filtered_flights, x="arr_delay", nbins=50, title="Arrival Delay Distribution")
st.plotly_chart(fig_delay)

# Delay vs time of day
st.subheader("Departure Delay vs Hour of Day")
filtered_flights["dep_hour"] = (filtered_flights["dep_time"] // 100).fillna(0).astype(int)
fig_hourly_delay = px.box(filtered_flights, x="dep_hour", y="dep_delay", points="all")
st.plotly_chart(fig_hourly_delay)

# Delay vs weather
st.subheader("Arrival Delay vs Wind Speed (Weather Analysis)")
weather_avg = weather.groupby("origin").agg({"wind_speed": "mean"}).reset_index()
weather_avg.rename(columns={"origin": "dest"}, inplace=True)
merged = pd.merge(filtered_flights, weather_avg, on="dest", how="left")
fig_weather = px.scatter(merged, x="wind_speed", y="arr_delay", trendline="ols", title="Delay vs Wind Speed")
st.plotly_chart(fig_weather)

# Statistics per route
st.subheader("Statistics per route")
if not filtered_flights.empty:
    route_stats = filtered_flights.groupby(["origin", "dest"]).agg({
        "flight": "count",
        "arr_delay": "mean",
        "air_time": "mean"
    }).reset_index()
    st.dataframe(route_stats)

# Bonus: Most used aircraft manufacturers to selected destinations
if destinations:
    st.subheader("Top aircraft manufacturers to selected destinations")
    dest_flights = filtered_flights[filtered_flights["dest"].isin(destinations)]
    merged_planes = pd.merge(dest_flights, planes, on="tailnum", how="left")
    manufacturer_counts = merged_planes["manufacturer"].value_counts().head(5).reset_index()
    manufacturer_counts.columns = ["Manufacturer", "Number of Flights"]
    st.dataframe(manufacturer_counts)

conn.close()
