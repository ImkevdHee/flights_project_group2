import sqlite3
import pandas as pd
import numpy as np
from timezonefinder import TimezoneFinder
from datetime import datetime, timedelta
import pytz

db_path = "flights_database.db"
conn = sqlite3.connect(db_path)

query = "SELECT * FROM flights"
flights_df = pd.read_sql(query, conn)

# Missing values
missing_values = flights_df.isnull().sum()
print("Missing values in each column before handling:", missing_values)

numerical_cols = flights_df.select_dtypes(include=['float64', 'int64']).columns
flights_df[numerical_cols] = flights_df[numerical_cols].fillna(0)

categorical_cols = flights_df.select_dtypes(include=['object']).columns
flights_df[categorical_cols] = flights_df[categorical_cols].fillna('UNKNOWN')

flights_df['dep_time'].fillna(flights_df['sched_dep_time'], inplace=True)
flights_df['arr_time'].fillna(flights_df['sched_arr_time'], inplace=True)


# Handle duplicates
duplicates_df = flights_df[flights_df.duplicated(subset=['year', 'month', 'day', 'sched_dep_time', 'carrier', 'flight', 'origin', 'dest'], keep=False)]
print(f"\nTotal duplicate flights found: {duplicates_df.shape[0]}")
flights_df = flights_df.drop_duplicates(subset=['year', 'month', 'day', 'sched_dep_time', 'carrier', 'flight', 'origin', 'dest'])

flights_df.to_sql("flights_cleaned", conn, if_exists="replace", index=False)

# Datetime
flights_df['sched_dep_time'] = pd.to_datetime(flights_df['sched_dep_time'], format='%H%M', errors='coerce')
flights_df['dep_time'] = pd.to_datetime(flights_df['dep_time'], format='%H%M', errors='coerce')
flights_df['sched_arr_time'] = pd.to_datetime(flights_df['sched_arr_time'], format='%H%M', errors='coerce')
flights_df['arr_time'] = pd.to_datetime(flights_df['arr_time'], format='%H%M', errors='coerce')

flights_df['arr_time'].fillna(flights_df['sched_arr_time'], inplace=True)
flights_df['sched_arr_time'].fillna(flights_df['sched_dep_time'] + pd.to_timedelta(flights_df['air_time'], unit='m'), inplace=True)

print(f"Missing air_time before imputation: {flights_df['air_time'].isnull().sum()}")

# air_time using actual departure and arrival times
flights_df.loc[flights_df['air_time'].isnull(), 'air_time'] = (
    (flights_df['arr_time'] - flights_df['dep_time']).dt.total_seconds() / 60
)

# If air_time is still missing, use scheduled times
flights_df.loc[flights_df['air_time'].isnull(), 'air_time'] = (
    (flights_df['sched_arr_time'] - flights_df['sched_dep_time']).dt.total_seconds() / 60
)

# Unrealistic air_time values
flights_df = flights_df[(flights_df['air_time'] >= 1) & (flights_df['air_time'] <= 1440)]

print(f"Missing air_time after imputation: {flights_df['air_time'].isnull().sum()}")

def order_flight_data(flights_df):
    flights_df.loc[flights_df['dep_time'] < flights_df['sched_dep_time'], 'dep_time'] = flights_df['sched_dep_time']
    flights_df.loc[flights_df['arr_time'] < flights_df['dep_time'], 'arr_time'] = flights_df['dep_time']
    flights_df['air_time'] = (flights_df['arr_time'] - flights_df['dep_time']).dt.total_seconds() / 60
    flights_df = flights_df[(flights_df['air_time'] >= 1) & (flights_df['air_time'] <= 1440)]
    return flights_df

flights_df = order_flight_data(flights_df)

# Local time
airport_coordinates = {
    'EWR': (40.6895, -74.1745),
    'SMF': (38.6954, -121.5914),
    'JFK': (40.6413, -73.7781),
    'ATL': (33.6407, -84.4279),
}

timezone_cache = {}
tf = TimezoneFinder()

def get_airport_timezone(airport_code):
    if airport_code in timezone_cache:
        return timezone_cache[airport_code]
    coords = airport_coordinates.get(airport_code)
    if coords:
        timezone_str = tf.timezone_at(lng=coords[1], lat=coords[0])
        if timezone_str:
            timezone_cache[airport_code] = pytz.timezone(timezone_str)
            return timezone_cache[airport_code]
    timezone_cache[airport_code] = pytz.UTC
    return pytz.UTC

def convert_to_local_arrival(row):
    if pd.isna(row['arr_time']):
        return None
    dest_tz = get_airport_timezone(row['dest'])
    return row['arr_time'].tz_localize('UTC').astimezone(dest_tz)

flights_df['local_arrival_time'] = flights_df.apply(convert_to_local_arrival, axis=1)

print("\nFirst few rows of cleaned data:")
print(flights_df.head())

cursor = conn.cursor()
query = """
SELECT p.type, 
       AVG(w.wind_speed) AS avg_wind_speed, 
       AVG(w.precip) AS avg_precipitation, 
       COUNT(f.flight) AS num_flights
FROM flights_cleaned f
JOIN weather w ON f.origin = w.origin AND f.time_hour = w.time_hour
JOIN planes p ON f.tailnum = p.tailnum
GROUP BY p.type
ORDER BY num_flights DESC;
"""

# Execute query and load results into a DataFrame
df = pd.read_sql_query(query, conn)
print(df)
df.fillna(0, inplace=True)
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel("Plane Type")
ax1.set_ylabel("Avg Wind Speed (mph)", color='tab:blue')
ax1.bar(df["type"], df["avg_wind_speed"], color='tab:blue', alpha=0.7, label="Avg Wind Speed")
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel("Avg Precipitation (inches)", color='tab:orange')
ax2.plot(df["type"], df["avg_precipitation"], color='tab:orange', marker='o', label="Avg Precipitation")
ax2.tick_params(axis='y', labelcolor='tab:orange')

fig.suptitle("Effect of Wind & Precipitation on Different Plane Types")
plt.show()



conn.close()
