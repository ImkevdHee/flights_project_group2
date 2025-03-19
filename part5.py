# Imported Libraries
import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt



# Paths
image_path = "airport_logo.jpg"
data_path = "flights_processed.csv"



# Constants
MIN_DATE = datetime.date(2023, 1, 1)              #YYYY/MM/DD
MAX_DATE = datetime.date(2023, 12, 31)            #YYYY/MM/DD
DATE_COLUMNS = 2
OUTER_NUMERICAL_COLUMNS = 3
INNER_NUMERICAL_COLUMNS = 2
LOGO_WIDTH = 250
HUNDRED_PERCENT = 100





# Variables
default_departure_airport = ['LGA', 'EWR', "JFK"]
default_destination_airport = ['BOS', 'ORD', 'MCO']
default_start_date = datetime.date(2023, 1, 1)    #YYYY/MM/DD
default_end_date = datetime.date(2023, 12, 31)    #YYYY/MM/DD
number_decimals_average_delay = 1
number_decimals_percentage_delay = 1


# Page Customization
st.set_page_config(layout="wide")


# ---------------------------------- DATA PREPARATION ---------------------------------- #

# Load the DataFrame
df = pd.read_csv(data_path)



# Convert Objects Back to Datetime
df['sched_dep_datetime'] = pd.to_datetime(df['sched_dep_datetime'])
df['dep_datetime'] = pd.to_datetime(df['dep_datetime'])
df['sched_arr_datetime'] = pd.to_datetime(df['sched_arr_datetime'])
df['arr_datetime'] = pd.to_datetime(df['arr_datetime'])




# ------------------------------------- FUNCTIONS -------------------------------------- #

# General Statistics Function
def get_flight_statistics(df):

    if len(df) == 0:
        return '-', '-', '-', '-'

    total_flights = len(df)
    
    unique_destinations = df['dest'].nunique()
    
    most_frequent_destination = df['dest'].value_counts().idxmax()
    most_frequent_count = df['dest'].value_counts().max()

    return total_flights, unique_destinations, most_frequent_destination, most_frequent_count


# Average Departure Delay
def get_average_departure_delay(df):

    if len(df) == 0:
        return '-'

    return round(df['dep_delay'].mean(), number_decimals_average_delay)


# Percentage Delayed vs On Time
def get_percentage_delayed(df):

    total = len(df)
    delayed = sum(df['dep_delay'] != 0)
    on_time = sum(df['dep_delay'] == 0)

    percentage_delayed = round(float(delayed)/total * HUNDRED_PERCENT, number_decimals_percentage_delay)
    percentage_on_time = round(float(on_time)/total * HUNDRED_PERCENT, number_decimals_percentage_delay)

    return percentage_delayed, percentage_on_time
    

    

# -------------------------------------- SIDEBAR --------------------------------------- #

# Sidebar Logo
st.sidebar.image(image_path, width=LOGO_WIDTH)



# Sidebar Airport Selection
st.sidebar.title("Airport Selection")
departure_airports = st.sidebar.multiselect("Select Departure Airports", df.origin.unique(), default=default_departure_airport)
arrival_airports = st.sidebar.multiselect("Select Destinations", df.dest.unique(), default=default_destination_airport)



# ------------------------------------- MAIN PAGE -------------------------------------- #

# Container Title
title_container = st.container()
with title_container:
    st.title("NYC Flights Dashboard")



# Container Date Selection
top_date_selection = st.container()

col1, col2 = st.columns(DATE_COLUMNS)

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



# Adjust DataFrame to Chosen Destinations and Dates
adjusted_df = df.loc[
                     df['origin'].isin(departure_airports) &
                    (df['dest'].isin(arrival_airports)) &
                    (df['dep_datetime'].dt.date > start_date) &
                    (df['dep_datetime'].dt.date <= end_date)
                     ]



# Container Numerical Statistics
total_flights, _, most_frequent_destination, most_frequent_count = get_flight_statistics(adjusted_df)
average_departure_delay = get_average_departure_delay(adjusted_df)

numerical_statistics_container = st.container()
col1, col2, col3 = st.columns(OUTER_NUMERICAL_COLUMNS, gap='medium', border=True)

with numerical_statistics_container:

    with col1:
        col11, col12 = st.columns(INNER_NUMERICAL_COLUMNS)
        with col11:
            st.header(total_flights)
        with col12:
            st.write("Total number of flights")

    with col2:
        col21, col22= st.columns(INNER_NUMERICAL_COLUMNS)
        with col21:
            st.header(average_departure_delay)
        with col22:
            st.write("Minutes of average departure delay")

    with col3:
        col31, col32 = st.columns(INNER_NUMERICAL_COLUMNS)
        with col31:
            st.header(f"{most_frequent_count}")
        with col32:
            st.write(f"Flights to **{most_frequent_destination}**")
            st.write("Most frequent destination")


# Container Graphical Statistics

delayed_percentage, on_time_percentage = get_percentage_delayed(adjusted_df)

graphical_statistics_container = st.container()
col1, col2, col3 = st.columns(3)
with col1:
    labels = ['Delayed', "On time"]
    explosion_value = 0.01
    fig1, ax1 = plt.subplots()

    ax1.pie([delayed_percentage, on_time_percentage],
            labels=labels,
            explode=[explosion_value,explosion_value],
            autopct="%1.1f%%",
            colors=['gray', 'blue'],
            radius=1,
            #textprops={'size':'smaller'}
            )
    col1.pyplot(fig1)
