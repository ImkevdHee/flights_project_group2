# Imported Libraries
import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns



# Paths
image_path = "airport_logo.jpg"
data_path = "flights_processed.csv"



# Constants
MIN_DATE = datetime.date(2023, 1, 1)              #YYYY/MM/DD
MAX_DATE = datetime.date(2023, 12, 31)            #YYYY/MM/DD
DATE_COLUMNS = 3
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
plot_label_fontsize = 12
plot_title_fontsize = 14
arrival_dist_bins = 30



# Page Customization
st.set_page_config(layout="wide")



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

    if len(df) == 0:
        return '-', '-'

    total = len(df)
    delayed = sum(df['dep_delay'] != 0)
    on_time = sum(df['dep_delay'] == 0)

    percentage_delayed = round(float(delayed)/total * HUNDRED_PERCENT, number_decimals_percentage_delay)
    percentage_on_time = round(float(on_time)/total * HUNDRED_PERCENT, number_decimals_percentage_delay)

    return percentage_delayed, percentage_on_time



# Data for Arrival Distribution
def get_data_for_arrival_distribution(df):
    
    if len(df) == 0:
        return '-'

    data = df[df['arr_delay'] < df['arr_delay'].quantile(0.99)] # Drop last percentile to exclude outliers
    data = data['arr_delay']
    return data

# Categorize Part of Day
def categorize_time_of_day(hour):
    if 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'



# ---------------------------------- DATA PREPARATION ---------------------------------- #

# Load the DataFrame
df = pd.read_csv(data_path)



# Convert Objects Back to Datetime
df['sched_dep_datetime'] = pd.to_datetime(df['sched_dep_datetime'])
df['dep_datetime'] = pd.to_datetime(df['dep_datetime'])
df['sched_arr_datetime'] = pd.to_datetime(df['sched_arr_datetime'])
df['arr_datetime'] = pd.to_datetime(df['arr_datetime'])



# Add Part of Day Column to DataFrame
df['part_of_day'] = df['hour'].apply(categorize_time_of_day)

    

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

col1, col2, col3 = st.columns(DATE_COLUMNS)

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
    with col3:
        part_of_day = st.selectbox("Select Part of Day",
                                   options=["Anytime", "Night", "Morning", "Afternoon", "Evening"])



# ---------------------------------- NUMERICAL STATISTICS ----------------------------------- #

# Adjust DataFrame to Chosen Destinations, Dates and Part of Day
adjusted_df = df.loc[
                     df['origin'].isin(departure_airports) &
                    (df['dest'].isin(arrival_airports)) &
                    (df['dep_datetime'].dt.date >= start_date) &
                    (df['dep_datetime'].dt.date <= end_date)
                    ]
if part_of_day != "Anytime":
    adjusted_df = adjusted_df.loc[df['part_of_day'] == part_of_day]



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



# ---------------------------------- GRAPHICAL STATISTICS ----------------------------------- #

# Container Graphical Statistics
graphical_statistics_container = st.container()
col1, col2 = st.columns([2,1],border=True)



# Pie Chart
delayed_percentage, on_time_percentage = get_percentage_delayed(adjusted_df)


if not isinstance(delayed_percentage, float):
    col2.write("No flights found")
else:
    with col2:
        labels = ['Delayed', "On time"]
        explosion_values = [0.03, 0]
        sns.set_theme(style="whitegrid")
        fig1, ax1 = plt.subplots(figsize=(4,4))

        ax1.pie([delayed_percentage, on_time_percentage],
                labels=labels,
                explode=explosion_values,
                autopct="%1.1f%%",
                colors=['lightgray', 'dodgerblue'],
                radius=0.8,
                wedgeprops={'edgecolor': 'black'},
                textprops={'fontsize': plot_label_fontsize,}
                )
        ax1.set_title("Flight Departures Delayed vs On Time",
                      fontsize=plot_title_fontsize,
                      fontweight='bold')
        col2.pyplot(fig1)



# Arrival Distribution
arrival_data = get_data_for_arrival_distribution(adjusted_df)

if isinstance(arrival_data, str):
    col1.write("No flights found")
else:
    with col1:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.histplot(arrival_data, 
                     ax=ax2, 
                     bins=arrival_dist_bins, 
                     color='royalblue',
                     alpha=0.7,
                     kde=True
                     )
        ax2.set_xlabel("Arrival Delay (minutes)", fontsize=plot_label_fontsize)
        ax2.set_ylabel("Frequency", fontsize=plot_label_fontsize)
        ax2.set_title("Arrival Delay Distribution", 
                      fontsize=plot_title_fontsize,
                      fontweight='bold'
                      )
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        col1.pyplot(fig2)
        
