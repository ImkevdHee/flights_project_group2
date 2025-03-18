# Imported Libraries
import streamlit as st
import pandas as pd
import datetime



# Constants
MIN_DATE = datetime.date(2023, 1, 1)              #YYYY/MM/DD
MAX_DATE = datetime.date(2023, 12, 31)            #YYYY/MM/DD



# Variables
default_departure_airport = "JFK"
default_destination_airport = "BOS"
default_start_date = datetime.date(2023, 1, 1)    #YYYY/MM/DD
default_end_date = datetime.date(2023, 12, 31)    #YYYY/MM/DD



# Load the DataFrame
df = pd.read_csv("flights_processed.csv")



# Convert objects back to datetime
df['sched_dep_datetime'] = pd.to_datetime(df['sched_dep_datetime'])
df['dep_datetime'] = pd.to_datetime(df['dep_datetime'])
df['sched_arr_datetime'] = pd.to_datetime(df['sched_arr_datetime'])
df['arr_datetime'] = pd.to_datetime(df['arr_datetime'])



# General Statistics Function
def get_flight_statistics(df):

    if len(df) == 0:
        return None, None, None, "No"

    total_flights = len(df)
    
    unique_destinations = df['dest'].nunique()
    
    most_frequent_destination = df['dest'].value_counts().idxmax()
    most_frequent_count = df['dest'].value_counts().max()

    return total_flights, unique_destinations, most_frequent_destination, most_frequent_count
    


# Sidebar Airport Selection
st.sidebar.title("Airport Selection")
departure_airports = st.sidebar.multiselect("Select Departure Airports", df.origin.unique(),default=default_departure_airport)
arrival_airports = st.sidebar.multiselect("Select Destinations", df.dest.unique(),default=default_destination_airport)



# Top Container Date Selection
top_date_selection = st.container()

col1, col2 = st.columns(2)

with top_date_selection:
    with col1:
        start_date = st.date_input("Select Start Date", 
                                   value=default_start_date,
                                   min_value=MIN_DATE,
                                   max_value=MAX_DATE)
    with col2:
        end_date = st.date_input("Select End Date",
                                 value=default_end_date,
                                 min_value=MIN_DATE,
                                 max_value=MAX_DATE)



# Adjust DataFrame to chosen destinations and dates
adjusted_df = df.loc[df['origin'].isin(departure_airports)]
adjusted_df = adjusted_df.loc[df['dest'].isin(arrival_airports)]
adjusted_df = adjusted_df.loc[df['dep_datetime'].dt.date > start_date]
adjusted_df = adjusted_df.loc[df['dep_datetime'].dt.date <= end_date]



# General Statistics
total_flights, unique_destinations, most_frequent_destination, most_frequent_count = get_flight_statistics(adjusted_df)

st.title("NYC Flights Dashboard")
st.header("info")
st.write(f"Statistics for the flights on this day:")
st.write(f"Total number of flights: {total_flights}")
st.write(f"Number of unique destinations: {unique_destinations}")
st.write(f"Most frequent destination: {most_frequent_destination} ({most_frequent_count} flights)")
