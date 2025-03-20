import sqlite3
import pandas as pd
import numpy as np
from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
from datetime import timedelta




db_path = "flights_database.db"
conn = sqlite3.connect(db_path)

query = "SELECT * FROM flights"
flights_df = pd.read_sql(query, conn)

# Step 1: Check for missing values in the dataset
missing_values = flights_df.isnull().sum()
print("Missing values in each column before handling:")
print(missing_values)

# Step 2: Handle missing values (Impute them)
# For simplicity, let's use different strategies depending on the column type

# Numerical columns (we'll fill with 0 or the median)
numerical_cols = flights_df.select_dtypes(include=['float64', 'int64']).columns
flights_df[numerical_cols] = flights_df[numerical_cols].fillna(0)

# Categorical columns (we'll fill with 'UNKNOWN')
categorical_cols = flights_df.select_dtypes(include=['object']).columns
flights_df[categorical_cols] = flights_df[categorical_cols].fillna('UNKNOWN')

# Datetime columns: We'll fill with the scheduled time or the most logical value
flights_df['dep_time'].fillna(flights_df['sched_dep_time'], inplace=True)
flights_df['arr_time'].fillna(flights_df['sched_arr_time'], inplace=True)

# Step 3: Check for duplicates
# We will look for exact duplicates based on the key columns:
duplicates_df = flights_df[flights_df.duplicated(subset=['year', 'month', 'day', 'sched_dep_time', 'carrier', 'flight', 'origin', 'dest'], keep=False)]

# Display the duplicates
print(f"\nTotal duplicate flights found: {duplicates_df.shape[0]}")
print(duplicates_df)

# Step 4: Drop duplicates (keeping the first occurrence)
flights_df = flights_df.drop_duplicates(subset=['year', 'month', 'day', 'sched_dep_time', 'carrier', 'flight', 'origin', 'dest'])

# Check the result
print("\nData after removing duplicates:")
print(flights_df.head())

# Step 5: Save the cleaned data back to the database
flights_df.to_sql("flights_cleaned", conn, if_exists="replace", index=False)

# After imputing missing values and removing duplicates, check again for missing values
missing_values_after = flights_df.isnull().sum()

# Print the result to see if any columns still have missing values
print("\nMissing values after handling:")
print(missing_values_after)

# If there are no missing values, print confirmation
if missing_values_after.sum() == 0:
    print("\nThere are no missing values remaining in the dataset!")
else:
    print("\nThere are still missing values in the dataset.")

# Convert scheduled and actual times to datetime objects
flights_df['sched_dep_time'] = pd.to_datetime(flights_df['sched_dep_time'], format='%H%M', errors='coerce')
flights_df['dep_time'] = pd.to_datetime(flights_df['dep_time'], format='%H%M', errors='coerce')
flights_df['sched_arr_time'] = pd.to_datetime(flights_df['sched_arr_time'], format='%H%M', errors='coerce')
flights_df['arr_time'] = pd.to_datetime(flights_df['arr_time'], format='%H%M', errors='coerce')

# Check for missing datetime values after conversion
missing_datetime_values = flights_df[['sched_dep_time', 'dep_time', 'sched_arr_time', 'arr_time']].isna().sum()
print("\nMissing datetime values after conversion:")
print(missing_datetime_values)

# Fill missing datetime values logically
flights_df['arr_time'].fillna(flights_df['sched_arr_time'], inplace=True)
flights_df['sched_arr_time'].fillna(flights_df['sched_dep_time'] + pd.to_timedelta(flights_df['air_time'], unit='m'), inplace=True)

# Final missing values check after filling datetime values
missing_values_after_final = flights_df.isnull().sum()
print("\nFinal missing values check after filling NaT values:")
print(missing_values_after_final)

# Save the cleaned data back to the database
flights_df.to_sql("flights_cleaned", conn, if_exists="replace", index=False)

# Optionally, check the first few rows after cleaning
print("\nFirst few rows of the cleaned data:")
print(flights_df.head())


# Inspect rows where dep_time is missing
missing_dep_time = flights_df[flights_df['dep_time'].isnull()]
print(f"Rows with missing dep_time: {missing_dep_time[['year', 'month', 'day', 'sched_dep_time', 'dep_time']]}")

# Inspect rows where arr_time is missing
missing_arr_time = flights_df[flights_df['arr_time'].isnull()]
print(f"Rows with missing arr_time: {missing_arr_time[['year', 'month', 'day', 'sched_arr_time', 'arr_time']]}")

# Inspect rows where sched_dep_time and sched_arr_time are missing
missing_sched_times = flights_df[flights_df['sched_dep_time'].isnull() & flights_df['sched_arr_time'].isnull()]
print(f"Rows with missing sched_dep_time and sched_arr_time: {missing_sched_times[['year', 'month', 'day', 'sched_dep_time', 'sched_arr_time']]}")

# Check for rows where air_time is also missing (for those that might need imputation based on air_time)
missing_air_time = flights_df[flights_df['air_time'].isnull()]
print(f"Rows with missing air_time: {missing_air_time[['year', 'month', 'day', 'air_time']]}")

# Fill dep_time with sched_dep_time where dep_time is missing
flights_df['dep_time'].fillna(flights_df['sched_dep_time'], inplace=True)
# Fill arr_time with sched_arr_time where arr_time is missing
flights_df['arr_time'].fillna(flights_df['sched_arr_time'], inplace=True)
# Fill missing arr_time based on sched_dep_time and air_time (if sched_dep_time and air_time are present)
flights_df['arr_time'].fillna(
    flights_df['sched_dep_time'] + pd.to_timedelta(flights_df['air_time'], unit='m'),
    inplace=True
)
# Final missing values check after applying imputation
missing_values_after_fix = flights_df.isnull().sum()
print("\nFinal missing values check after imputation:")
print(missing_values_after_fix)


def order_flight_data(flights_df):
    # Ensure dep_time is not earlier than sched_dep_time
    flights_df.loc[flights_df['dep_time'] < flights_df['sched_dep_time'], 'dep_time'] = flights_df['sched_dep_time']
    
    # Ensure arr_time is not earlier than dep_time
    flights_df.loc[flights_df['arr_time'] < flights_df['dep_time'], 'arr_time'] = flights_df['dep_time']
    
    # Recalculate air_time based on the difference between arr_time and dep_time
    flights_df['air_time'] = (flights_df['arr_time'] - flights_df['dep_time']).dt.total_seconds() / 60
    
    # Remove unreasonable air_time values (less than 1 minute or more than 24 hours)
    flights_df = flights_df[(flights_df['air_time'] >= 1) & (flights_df['air_time'] <= 1440)]
    
    return flights_df

# Example usage:
# Assuming flights_df is your dataframe containing flight data
flights_df = order_flight_data(flights_df)

# Display the ordered data
print(flights_df.head())




# Example dictionary mapping airports to their respective latitudes and longitudes
airport_coordinates = {
    'EWR': (40.6895, -74.1745),
    'SMF': (38.6954, -121.5914),
    'JFK': (40.6413, -73.7781),
    'ATL': (33.6407, -84.4279),
    # Add more airports with their latitudes and longitudes
}

# Function to get the timezone for an airport code using timezonefinder
def get_airport_timezone(airport_code):
    # Get the latitude and longitude for the airport
    coords = airport_coordinates.get(airport_code)
    
    if not coords:
        # If the airport coordinates are not in the dictionary, return UTC
        return pytz.UTC
    
    lat, lon = coords
    
    # Initialize timezonefinder object
    tf = TimezoneFinder()

    # Use the timezonefinder to get the timezone at the given coordinates
    timezone_str = tf.timezone_at(lng=lon, lat=lat)

    # Return the timezone object
    if timezone_str:
        return pytz.timezone(timezone_str)
    else:
        return pytz.UTC  # Default to UTC if timezone is not found

# Function to add local arrival time to flights dataframe
def add_local_arrival_time(flights_df):
    # Create a new column for the local arrival time
    local_arrival_times = []

    for _, row in flights_df.iterrows():
        # Get the timezone for the destination airport
        dest_tz = get_airport_timezone(row['dest'])

        # Convert arrival time to UTC first (assuming arr_time is in UTC)
        arr_time_utc = row['arr_time'].tz_localize('UTC')

        # Convert arrival time to the destination timezone
        arr_time_local = arr_time_utc.astimezone(dest_tz)

        # Append the local arrival time to the list
        local_arrival_times.append(arr_time_local)

    # Add the local arrival times as a new column in the DataFrame
    flights_df['local_arrival_time'] = local_arrival_times
    return flights_df

# Example usage of the function:
flights_df = add_local_arrival_time(flights_df)

# Check the first few rows to see the new local arrival time column
print(flights_df[['year', 'month', 'day', 'origin', 'dest', 'arr_time', 'local_arrival_time']].head())


conn.close()



