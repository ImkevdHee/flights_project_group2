import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


# Connect to the SQLite database
db_path = "flights_database.db"
conn = sqlite3.connect(db_path)

# Load the flights table into a Pandas DataFrame
query = "SELECT * FROM flights"
flights_df = pd.read_sql(query, conn)

# Check for missing values
missing_values = flights_df.isnull().sum()

# Print the count of missing values per column
print("Missing values in each column:")
print(missing_values)


flights_df['dep_time'].fillna(-1, inplace=True)
flights_df['dep_delay'].fillna(-1, inplace=True)
flights_df['arr_time'].fillna(-1, inplace=True)
flights_df['arr_delay'].fillna(-1, inplace=True)
flights_df['tailnum'].fillna("UNKNOWN", inplace=True)
flights_df['air_time'].fillna(-1, inplace=True)

# Save the cleaned table (replace old one)
flights_df.to_sql("flights_cleaned", conn, if_exists="replace", index=False)

# Identify duplicate flights based on key columns
duplicate_flights = flights_df.duplicated(subset=['year', 'month', 'day', 'sched_dep_time', 'carrier', 'flight', 'origin', 'dest'], keep=False)

# Filter and display only duplicate rows
duplicates_df = flights_df[duplicate_flights]

# Count duplicates
num_duplicates = duplicates_df.shape[0]

print(f"Total duplicate flights found: {num_duplicates}")
print(duplicates_df)
flights_df = flights_df.drop_duplicates(subset=['year', 'month', 'day', 'sched_dep_time', 'carrier', 'flight', 'origin', 'dest'])









# Query to get all the data from the flights table
query = "SELECT * FROM flights;"
flights_df = pd.read_sql(query, conn)



def convert_to_datetime(time_value, year, month, day):
    if pd.isna(time_value) or time_value == 0:
        return None  # Handle NaN or zero values (invalid time)
    
    # Ensure the time_value is an integer
    time_value = int(time_value)  # Force conversion to integer

    # If time is 2400, treat it as 0000 (midnight)
    if time_value == 2400:
        time_value = 0
    
    # Extract hours and minutes
    hours = time_value // 100
    minutes = time_value % 100

    # Ensure hour is within valid range (0-23)
    if hours >= 24:
        hours = 23  # Cap hours at 23

    # Return datetime object
    return datetime(year, month, day, hours, minutes)

# Convert the scheduled and actual times to datetime objects
flights_df['sched_dep_datetime'] = flights_df.apply(
    lambda row: convert_to_datetime(row['sched_dep_time'], row['year'], row['month'], row['day']), axis=1)
flights_df['dep_datetime'] = flights_df.apply(
    lambda row: convert_to_datetime(row['dep_time'], row['year'], row['month'], row['day']), axis=1)
flights_df['sched_arr_datetime'] = flights_df.apply(
    lambda row: convert_to_datetime(row['sched_arr_time'], row['year'], row['month'], row['day']), axis=1)
flights_df['arr_datetime'] = flights_df.apply(
    lambda row: convert_to_datetime(row['arr_time'], row['year'], row['month'], row['day']), axis=1)

# Print the DataFrame to verify the conversion
print(flights_df[['sched_dep_datetime', 'dep_datetime', 'sched_arr_datetime', 'arr_datetime']].head())




conn.close()