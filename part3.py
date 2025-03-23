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

db_path = "flights_database.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

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
R = 6378

def geodesic_distance(lat1, lon1, lat2, lon2, R):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    
    delta_phi = lat2 - lat1
    delta_lambda = lon2 - lon1
    phi_m = (lat1 + lat2) / 2.0
    
    term1 = ((2 * np.sin(delta_phi / 2)) * np.cos(delta_lambda / 2)) ** 2
    term2 = (2 * np.cos(phi_m) * np.sin(delta_lambda / 2)) ** 2
    
    return R * np.sqrt(term1 + term2)

errors = []

for actual_distance, lat1, lon1, lat2, lon2 in data:
    computed_distance = geodesic_distance(lat1, lon1, lat2, lon2, R)
computed_distance_miles = computed_distance * 0.621371

error = abs(computed_distance_miles - actual_distance)
errors.append(error)

avg_error = sum(errors) / len(errors)
max_error = max(errors)
min_error = min(errors)

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

query = """
    SELECT DISTINCT origin 
    FROM flights;
"""
cur.execute(query)
origin_airports = cur.fetchall()

origin_airport_codes = [airport[0] for airport in origin_airports]

query_airports = f"""
    SELECT * 
    FROM airports 
    WHERE faa IN ({','.join('?' for _ in origin_airport_codes)});
"""
cur.execute(query_airports, origin_airport_codes)

airport_data = cur.fetchall()
columns = [description[0] for description in cur.description]  # Get column names
airports_df = pd.DataFrame(airport_data, columns=columns)

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
    total_flights = len(flights_df)
    
    unique_destinations = flights_df['destination'].nunique()
    
    most_frequent_destination = flights_df['destination'].value_counts().idxmax()
    most_frequent_count = flights_df['destination'].value_counts().max()

    print(f"Statistics for the flights on this day:")
    print(f"Total number of flights: {total_flights}")
    print(f"Number of unique destinations: {unique_destinations}")
    print(f"Most frequent destination: {most_frequent_destination} ({most_frequent_count} flights)")

get_input_date()


def get_plane_types_for_trajectory(departure_airport_faa, arrival_airport_faa):

    query = """
        SELECT p.type, COUNT(*) as count
        FROM flights f
        JOIN planes p ON f.tailnum = p.tailnum
        WHERE f.origin = ? AND f.dest = ?
        GROUP BY p.type
    """

    cur.execute(query, (departure_airport_faa, arrival_airport_faa))
    results = cur.fetchall()

    plane_types_count = {row[0]: row[1] for row in results}

    return plane_types_count

def get_input_airports():
    departure_airport_faa = input("Enter the FAA code of the departure airport: ").strip().upper()
    arrival_airport_faa = input("Enter the FAA code of the arrival airport: ").strip().upper()

    plane_types_count = get_plane_types_for_trajectory(departure_airport_faa, arrival_airport_faa)

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

def get_delayed_flights(destination, start_month, end_month):

    query = """
        SELECT COUNT(*) 
        FROM flights f
        WHERE f.dest = ? 
        AND f.dep_delay > 0 
        AND f.month BETWEEN ? AND ?;
    """
    
    cur.execute(query, (destination, start_month, end_month))
    
    delayed_flights_count = cur.fetchone()[0]
    
    
    return delayed_flights_count

def get_input_delay():
 
    destination = input("Enter destination FAA code: ").strip().upper()
    start_month = int(input("Enter start month (1-12): "))
    end_month = int(input("Enter end month (1-12 and has to be bigger than start month): "))
    
    delayed_flights = get_delayed_flights(destination, start_month, end_month)
    
    print(f"Delayed flights to {destination} from {start_month} to {end_month}: {delayed_flights}")

get_input_delay()

def get_top_airplane_manufacturers(destination):

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

def analyze_relationship():
    df = fetch_flight_data()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="distance", y="arr_delay", color='blue')
    plt.title("Relationship between Flight Distance and Arrival Delay")
    plt.xlabel("Flight Distance (miles)")
    plt.ylabel("Arrival Delay (minutes)")
    plt.grid(True)
    plt.show()

    correlation, _ = pearsonr(df["distance"], df["arr_delay"])
    print(f"Pearson Correlation Coefficient: {correlation:.2f}")

analyze_relationship()

# A Pearson Correlation Coefficient of 0.01 indicates that there is virtually no linear relationship between flight distance and arrival delay. 
# This suggests that the distance of a flight does not significantly impact its arrival delay time.


def update_planes_with_speed():

    query = """
    SELECT f.tailnum, AVG(f.distance / f.air_time) AS avg_speed
    FROM flights f
    WHERE f.air_time > 0  -- Make sure we don't divide by zero
    GROUP BY f.tailnum;
    """

    cur.execute(query)
    plane_speeds = cur.fetchall()

    for plane_speed in plane_speeds:
        tailnum = plane_speed[0]
        avg_speed = plane_speed[1]

        update_query = """
        UPDATE planes
        SET speed = ?
        WHERE tailnum = ?;
        """
        cur.execute(update_query, (avg_speed, tailnum))

    conn.commit()

update_planes_with_speed()

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the initial bearing (direction) from one point to another on the Earth.
    Returns the bearing in degrees.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    delta_lon = lon2 - lon1

    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))

    bearing = math.atan2(x, y)

    bearing = math.degrees(bearing)

    bearing = (bearing + 360) % 360

    return bearing

def get_flight_directions():
    nyc_airports = {
        "JFK": (40.6413, -73.7781),
        "LGA": (40.7769, -73.8740),
        "EWR": (40.6895, -74.1745)
    }

    query = "SELECT faa, lat, lon FROM airports WHERE faa NOT IN ('JFK', 'LGA', 'EWR');"
    cur.execute(query)
    airports = cur.fetchall()

    directions = {}

    for airport in airports:
        faa, lat, lon = airport
        bearings = []

        for nyc, (nyc_lat, nyc_lon) in nyc_airports.items():
            bearing = calculate_bearing(nyc_lat, nyc_lon, lat, lon)
            bearings.append(bearing)

        avg_bearing = sum(bearings) / len(bearings)
        directions[faa] = avg_bearing


    return directions

flight_directions = get_flight_directions()

for airport, direction in flight_directions.items():
    print(f"Flight direction to {airport}: {direction:.2f}°")


def compute_inner_product(flight_direction, wind_direction, wind_speed):
    """
    Compute the inner product between the flight direction and wind speed.
    :param flight_direction: Direction of the flight in degrees.
    :param wind_direction: Direction of the wind in degrees.
    :param wind_speed: Wind speed in knots.
    :return: Inner product value.
    """
    theta = math.radians(abs(flight_direction - wind_direction))

    inner_product = wind_speed * math.cos(theta)

    return inner_product

def get_flight_wind_inner_product(flight_id):
    """
    Retrieves flight direction, wind direction, and wind speed from the database 
    and computes the inner product.
    :param flight_id: The unique flight identifier.
    :return: Inner product value.
    """

    query_flight = """
        SELECT f.origin, f.dest, a.lat, a.lon
        FROM flights f
        JOIN airports a ON f.dest = a.faa
        WHERE f.flight = ?;
    """
    cur.execute(query_flight, (flight_id,))
    flight_data = cur.fetchone()

    if not flight_data:
        print(f"Flight {flight_id} not found.")
        return None

    origin, dest, lat, lon = flight_data

    nyc_airports = {
        "JFK": (40.6413, -73.7781),
        "LGA": (40.7769, -73.8740),
        "EWR": (40.6895, -74.1745)
    }

    if origin not in nyc_airports:
        print(f"Origin airport {origin} is not in NYC.")
        return None

    nyc_lat, nyc_lon = nyc_airports[origin]
    flight_direction = math.degrees(math.atan2(lon - nyc_lon, lat - nyc_lat)) % 360

    query_wind = """
        SELECT w.wind_dir, w.wind_speed
        FROM weather w
        WHERE w.origin = ? 
        ORDER BY w.time_hour DESC LIMIT 1;
    """
    cur.execute(query_wind, (origin,))
    wind_data = cur.fetchone()

    if not wind_data:
        print(f"No wind data found for airport {origin}.")
        return None

    wind_direction, wind_speed = wind_data

    inner_product = compute_inner_product(flight_direction, wind_direction, wind_speed)

    return inner_product

flight_id = input("Enter Flight ID: ")
result = get_flight_wind_inner_product(flight_id)

if result is not None:
    if result > 0:
        interpretation = "tailwind"
        expected_airtime = "shorter airtime"
    elif result < 0:
        interpretation = "headwind"
        expected_airtime = "longer airtime"

    else:
        interpretation = "crosswind"
        expected_airtime = "minimal impact on airtime"

    print(f"Inner product between flight direction and wind speed: {result:.2f}")
    print(f"This correlates to a {interpretation}")
    print(f"Therefore, we expect a {expected_airtime}")

conn.close()


