import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

db_path = "flights_database.db"
conn = sqlite3.connect(db_path)

# Load the flights data
query = "SELECT * FROM flights"
flights_df = pd.read_sql(query, conn)

# Drop rows where crucial columns have missing values (e.g., dep_time, arr_time, sched_dep_time)
flights_df.dropna(subset=['dep_time', 'arr_time', 'sched_dep_time'], inplace=True)


# After dropping rows with missing values, check if there are still any discrepancies
missing_values = flights_df.isnull().sum()
print("Missing values in each column after dropping rows:")
print(missing_values)

# Save the cleaned table (replace old one)
flights_df.to_sql("flights_cleaned", conn, if_exists="replace", index=False)

# Identify and remove duplicates based on key columns
duplicate_flights = flights_df.duplicated(subset=['year', 'month', 'day', 'sched_dep_time', 'carrier', 'flight', 'origin', 'dest'], keep=False)
flights_df = flights_df[~duplicate_flights]  # Remove duplicate rows

# Vectorized function to convert time to datetime
def convert_to_datetime(time_value, year, month, day):
    if pd.isna(time_value) or time_value == 0:
        return None
    
    time_value = int(time_value)
    if time_value == 2400:
        time_value = 0
    
    hours = time_value // 100
    minutes = time_value % 100
    hours = min(hours, 23)  # Ensure hours don't exceed 23
    return datetime(year, month, day, hours, minutes)

# Apply conversion for all date/time columns
flights_df['sched_dep_datetime'] = flights_df.apply(
    lambda row: convert_to_datetime(row['sched_dep_time'], row['year'], row['month'], row['day']), axis=1)
flights_df['dep_datetime'] = flights_df.apply(
    lambda row: convert_to_datetime(row['dep_time'], row['year'], row['month'], row['day']), axis=1)
flights_df['sched_arr_datetime'] = flights_df.apply(
    lambda row: convert_to_datetime(row['sched_arr_time'], row['year'], row['month'], row['day']), axis=1)
flights_df['arr_datetime'] = flights_df.apply(
    lambda row: convert_to_datetime(row['arr_time'], row['year'], row['month'], row['day']), axis=1)

# Efficiently check for discrepancies
def check_flight_data_order(flights_df):
    discrepancies = []

    # Check if dep_time is earlier than sched_dep_time
    mask_dep_time = flights_df['dep_time'] < flights_df['sched_dep_time']
    discrepancies += [f"Flight {row['flight']} (ID: {row['flight']}) has dep_time earlier than sched_dep_time."
                      for idx, row in flights_df[mask_dep_time].iterrows()]

    # Check if arr_time is earlier than dep_time
    mask_arr_time = flights_df['arr_time'] < flights_df['dep_time']
    discrepancies += [f"Flight {row['flight']} (ID: {row['flight']}) has arr_time earlier than dep_time."
                      for idx, row in flights_df[mask_arr_time].iterrows()]

    # Check if air_time matches the difference between arr_time and dep_time
    mask_air_time = abs(flights_df['air_time'] - (flights_df['arr_time'] - flights_df['dep_time'])) > 5  # Allowing a margin of 5 minutes
    discrepancies += [f"Flight {row['flight']} (ID: {row['flight']}) has inconsistent air_time. Expected: {row['arr_time'] - row['dep_time']}, Found: {row['air_time']}."
                      for idx, row in flights_df[mask_air_time].iterrows()]

    return discrepancies

# Fix arrival time discrepancies by setting to scheduled arrival time
def fix_arrival_times(flights_df):
    # Replace arr_time with sched_arr_time where arr_time is earlier than dep_time or missing
    mask_arr_time = (flights_df['arr_time'] < flights_df['dep_time']) | flights_df['arr_time'].isna()
    flights_df.loc[mask_arr_time, 'arr_time'] = flights_df.loc[mask_arr_time, 'sched_arr_time']
    return flights_df

# Fix departure time discrepancies by setting to scheduled departure time
def fix_departure_times(flights_df):
    mask_missing_dep_time = flights_df['dep_time'].isna()
    flights_df.loc[mask_missing_dep_time, 'dep_time'] = flights_df.loc[mask_missing_dep_time, 'sched_dep_time']
    return flights_df

# Recalculate air_time based on arr_time and dep_time
def recalculate_air_time(flights_df):
    mask_valid_times = pd.notna(flights_df['arr_time']) & pd.notna(flights_df['dep_time'])
    flights_df.loc[mask_valid_times, 'air_time'] = flights_df.loc[mask_valid_times, 'arr_time'] - flights_df.loc[mask_valid_times, 'dep_time']
    return flights_df

# Apply all fixes
flights_df = fix_arrival_times(flights_df)
flights_df = fix_departure_times(flights_df)
flights_df = recalculate_air_time(flights_df)

# Check for discrepancies again after fixes
discrepancies_after_fix = check_flight_data_order(flights_df)

# Output the results
if discrepancies_after_fix:
    print(f"Total discrepancies found: {len(discrepancies_after_fix)}")
    for discrepancy in discrepancies_after_fix:
        print(discrepancy)
else:
    print("All flights data are now in order!")

# Save the cleaned data after processing
flights_df.to_sql("flights_cleaned", conn, if_exists="replace", index=False)

conn.close()
