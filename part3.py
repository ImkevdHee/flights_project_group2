# Part 3
import sqlite3
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from collections import defaultdict
import seaborn as sns
from scipy.stats import pearsonr

# Connect to the SQLite database
db_path = "flights_database.db"  # Relative path since it's in the same folder
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Check if tables exist
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()

print("Tables in database:", tables)



# Query to get the distance, and lat/lon for origin and destination
query = """
    SELECT f.distance, a1.lat, a1.lon, a2.lat, a2.lon 
    FROM flights f
    JOIN airports a1 ON f.origin = a1.faa
    JOIN airports a2 ON f.dest = a2.faa
    LIMIT 100;  -- Limit to 100 rows for efficiency
"""
cur.execute(query)
data = cur.fetchall()





# Earth's radius in kilometers
R = 6378  # Radius of the Earth in km

# Geodesic distance calculation function
def geodesic_distance(lat1, lon1, lat2, lon2, R):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    
    # Formula components
    delta_phi = lat2 - lat1
    delta_lambda = lon2 - lon1
    phi_m = (lat1 + lat2) / 2.0
    
    term1 = ((2 * np.sin(delta_phi / 2)) * np.cos(delta_lambda / 2)) ** 2
    term2 = (2 * np.cos(phi_m) * np.sin(delta_lambda / 2)) ** 2
    
    return R * np.sqrt(term1 + term2)

# List to store error values
errors = []

# Loop over the data to calculate the error in distances
for actual_distance, lat1, lon1, lat2, lon2 in data:
    # Calculate geodesic distance from JFK (or the relevant origin/destination pair)
    computed_distance = geodesic_distance(lat1, lon1, lat2, lon2, R)
# Convert computed distances from km to miles
computed_distance_miles = computed_distance * 0.621371

# Compare with the database distance in miles
error = abs(computed_distance_miles - actual_distance)
errors.append(error)


# Calculate average, max, and min errors
avg_error = sum(errors) / len(errors)
max_error = max(errors)
min_error = min(errors)

# Output the results
print(f"Average error: {avg_error:.2f} km")
print(f"Max error: {max_error:.2f} km")
print(f"Min error: {min_error:.2f} km")


plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, edgecolor='black')
plt.title('Distribution of Errors Between Computed and Database Distances')
plt.xlabel('Error (miles)')
plt.ylabel('Frequency')
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# The small and consistent errors, indicated by the single bar in the histogram, suggest that the computed distances closely match the database values, demonstrating the accuracy and consistency of the distance calculations.


# Query to get distinct origin airports from flights
query = """
    SELECT DISTINCT origin 
    FROM flights;
"""
cur.execute(query)
origin_airports = cur.fetchall()

# Convert list of origin airports to a list of airport codes
origin_airport_codes = [airport[0] for airport in origin_airports]

# Now, get the details of those airports from the airports table
query_airports = f"""
    SELECT * 
    FROM airports 
    WHERE faa IN ({','.join('?' for _ in origin_airport_codes)});
"""
cur.execute(query_airports, origin_airport_codes)

# Fetch the results and convert them into a DataFrame
airport_data = cur.fetchall()
columns = [description[0] for description in cur.description]  # Get column names
airports_df = pd.DataFrame(airport_data, columns=columns)


# Display the DataFrame with NYC airports from which flights depart
print(airports_df)


for column in columns:
    print(column)


def plot_flight_destinations(month, day, airport_faa):

    query = """
        SELECT f.dest, a.lat, a.lon
        FROM flights f
        JOIN airports a ON f.dest = a.faa
        WHERE f.origin = ? AND f.month = ? AND f.day = ?
    """
    cur.execute(query, (airport_faa, month, day))
    flights = cur.fetchall()

    if not flights:
        print(f"No flights found from airport {airport_faa} on {month}/{day}.")
        conn.close()
        return

    flights_df = pd.DataFrame(flights, columns=['destination', 'lat', 'lon'])

    fig = px.scatter_geo(flights_df,
                         lat='lat',
                         lon='lon',
                         hover_name='destination',
                         title=f"Flight Destinations from {airport_faa} on {month}/{day}",
                         projection='natural earth')

    fig.show()
    get_flight_statistics(flights_df)

def get_input_date():
    month = int(input("Enter the month (1-12): "))
    day = int(input("Enter the day (1-31): "))
    airport_faa = input("Enter the FAA code of the airport: ").strip().upper()

    plot_flight_destinations(month, day, airport_faa)

def get_flight_statistics(flights_df):
    # Total number of flights
    total_flights = len(flights_df)
    
    # Number of unique destinations
    unique_destinations = flights_df['destination'].nunique()
    
    # Most frequent destination
    most_frequent_destination = flights_df['destination'].value_counts().idxmax()
    most_frequent_count = flights_df['destination'].value_counts().max()

    # Print the statistics
    print(f"Statistics for the flights on this day:")
    print(f"Total number of flights: {total_flights}")
    print(f"Number of unique destinations: {unique_destinations}")
    print(f"Most frequent destination: {most_frequent_destination} ({most_frequent_count} flights)")

get_input_date()




def get_plane_types_for_trajectory(departure_airport_faa, arrival_airport_faa):


    # Query to get the plane types for flights between the given departure and arrival airports
    query = """
        SELECT p.type, COUNT(*) as count
        FROM flights f
        JOIN planes p ON f.tailnum = p.tailnum
        WHERE f.origin = ? AND f.dest = ?
        GROUP BY p.type
    """

    # Execute the query with the provided airports as parameters
    cur.execute(query, (departure_airport_faa, arrival_airport_faa))
    results = cur.fetchall()


    # Convert the results into a dictionary
    plane_types_count = {row[0]: row[1] for row in results}

    return plane_types_count

def get_input_airports():
    # Get user input for departure and arrival airports (FAA codes)
    departure_airport_faa = input("Enter the FAA code of the departure airport: ").strip().upper()
    arrival_airport_faa = input("Enter the FAA code of the arrival airport: ").strip().upper()

    # Call the get_plane_types_for_trajectory function with user input
    plane_types_count = get_plane_types_for_trajectory(departure_airport_faa, arrival_airport_faa)

    # Print the results
    if plane_types_count:
        print("\nPlane types used on this route:")
        for plane_type, count in plane_types_count.items():
            print(f"{plane_type}: {count} flights")
    else:
        print(f"No data found for flights between {departure_airport_faa} and {arrival_airport_faa}.")


get_input_airports()



query = """
    SELECT a.name, AVG(f.dep_delay) AS avg_dep_delay
    FROM flights f
    JOIN airlines a ON f.carrier = a.carrier
    GROUP BY a.name
    ORDER BY avg_dep_delay DESC;
"""

cur.execute(query)
data = cur.fetchall()

df = pd.DataFrame(data, columns=['airline_name', 'avg_dep_delay'])

plt.figure(figsize=(12, 6)) 
plt.bar(df['airline_name'], df['avg_dep_delay'], color='skyblue')
plt.ylabel('Average Departure Delay (minutes)')
plt.title('Average Departure Delay per Airline')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()



# Function to get delayed flights
def get_delayed_flights(destination, start_month, end_month):

    # SQL query to count delayed flights to the specified destination and month range
    query = """
        SELECT COUNT(*) 
        FROM flights f
        WHERE f.dest = ? 
        AND f.dep_delay > 0 
        AND f.month BETWEEN ? AND ?;
    """
    
    # Execute the query with parameters: destination, start_month, and end_month
    cur.execute(query, (destination, start_month, end_month))
    
    # Fetch the result (the count of delayed flights)
    delayed_flights_count = cur.fetchone()[0]
    
    
    return delayed_flights_count

# Function to get user input
def get_input_delay():
 
    destination = input("Enter destination FAA code: ").strip().upper()
    start_month = int(input("Enter start month (1-12): "))
    end_month = int(input("Enter end month (1-12 and has to be bigger than start month): "))
    
    # Call the function to get delayed flights
    delayed_flights = get_delayed_flights(destination, start_month, end_month)
    
    # Print the result
    print(f"Delayed flights to {destination} from {start_month} to {end_month}: {delayed_flights}")

# Run the function to get user input and output the result
get_input_delay()


import sqlite3
import pandas as pd

# Function to get top 5 airplane manufacturers for flights to a specific destination
def get_top_airplane_manufacturers(destination):


    # SQL query to get the top 5 airplane manufacturers for flights to the specified destination
    query = """
        SELECT p.manufacturer, COUNT(f.tailnum) AS num_flights
        FROM flights f
        JOIN planes p ON f.tailnum = p.tailnum
        WHERE f.dest = ?
        GROUP BY p.manufacturer
        ORDER BY num_flights DESC
        LIMIT 5;
    """

    cur.execute(query, (destination,))
    data = cur.fetchall()


    if not data:
        return f"No flights found to destination {destination}"

    manufacturers_df = pd.DataFrame(data, columns=["Manufacturer", "Number of Flights"])

    return manufacturers_df

def get_input_manufacturers():
    destination = input("Enter the destination FAA code: ").strip().upper()
    
    result = get_top_airplane_manufacturers(destination)
    
    if isinstance(result, pd.DataFrame):
        print(f"Top 5 Airplane Manufacturers for flights to {destination}:")
        print(result)
    else:
        print(result)

get_input_manufacturers()




# Function to fetch data from the flights table
def fetch_flight_data():


    query = """
        SELECT f.distance, f.arr_delay
        FROM flights f
        WHERE f.distance IS NOT NULL AND f.arr_delay IS NOT NULL
    """
    cur.execute(query)
    data = cur.fetchall()

    
    df = pd.DataFrame(data, columns=["distance", "arr_delay"])
    return df

# Function to visualize the relationship and calculate correlation
def analyze_relationship():
    df = fetch_flight_data()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="distance", y="arr_delay", color='blue')
    plt.title("Relationship between Flight Distance and Arrival Delay")
    plt.xlabel("Flight Distance (miles)")
    plt.ylabel("Arrival Delay (minutes)")
    plt.grid(True)
    plt.show()

    # Calculate Pearson correlation coefficient
    correlation, _ = pearsonr(df["distance"], df["arr_delay"])
    print(f"Pearson Correlation Coefficient: {correlation:.2f}")

analyze_relationship()

# A Pearson Correlation Coefficient of 0.01 indicates that there is virtually no linear relationship between flight distance and arrival delay. This suggests that the distance of a flight does not significantly impact its arrival delay time.



import sqlite3

def update_planes_with_speed():


    # Query to get the average speed per plane model (tailnum)
    query = """
    SELECT f.tailnum, AVG(f.distance / f.air_time) AS avg_speed
    FROM flights f
    WHERE f.air_time > 0  -- Make sure we don't divide by zero
    GROUP BY f.tailnum;
    """

    cur.execute(query)
    plane_speeds = cur.fetchall()

    # Update the 'planes' table with the calculated speeds
    for plane_speed in plane_speeds:
        tailnum = plane_speed[0]
        avg_speed = plane_speed[1]

        # Update the speed in the planes table
        update_query = """
        UPDATE planes
        SET speed = ?
        WHERE tailnum = ?;
        """
        cur.execute(update_query, (avg_speed, tailnum))

    # Commit the changes and close the connection
    conn.commit()


# Call the function to update the planes table with the average speeds
update_planes_with_speed()





# Close the database connection
conn.close()


