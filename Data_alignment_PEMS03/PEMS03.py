# %%
# !tar -xzvf TrafficEvents_Aug16_Dec20_Publish.tar.gz

# %%
!ls -lah

# %%
import pandas as pd
import geopandas as gpd

from pytz import timezone
from datetime import datetime

from shapely.geometry import Point

# %% [markdown]
# # Map clusters

# %%
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import folium
from folium.plugins import FastMarkerCluster

# %%
m = folium.Map(location=[38.575764, -121.478851], zoom_start=10)


add_circle_markers(traffic_events_non_highway_gdf, m, 'red', popup_column='Street')
add_circle_markers(sensors_gdf, m, 'blue', popup_column='ID')

# Display the map
m

# %%


# %% [markdown]
# # Time and spatial clip

# %%
pacific_tz = timezone('US/Pacific')
utc_tz = timezone('UTC')

# %%
# csv_file = 'TrafficEvents_Aug16_Dec20_Publish.csv'
csv_file = 'TrafficEvents_Sep1_Nov30_Sacramento.csv'

# Define time range
start_time_local = '2018-09-01 00:00'
end_time_local = '2018-12-01 00:00'

start_time_utc = pacific_tz.localize(datetime.strptime(start_time_local, '%Y-%m-%d %H:%M')).astimezone(utc_tz).strftime('%Y-%m-%d %H:%M:%S')
end_time_utc = pacific_tz.localize(datetime.strptime(end_time_local, '%Y-%m-%d %H:%M')).astimezone(utc_tz).strftime('%Y-%m-%d %H:%M:%S')

# Define spatial range
lat_range = [38.375, 38.853]
long_range = [-121.735, -121.112]

# %%
filtered_chunks = []
# chunk size
chunksize = 10 ** 6

# Iterate over the CSV file in chunks
for chunk in pd.read_csv(csv_file, chunksize=chunksize):
    # Filter by time
    chunk_filtered_time = chunk[(chunk['EndTime(UTC)'] >= start_time_utc) & (chunk['StartTime(UTC)'] <= end_time_utc)]
    
    # Filter by spatial
    chunk_filtered = chunk_filtered_time[
        (chunk_filtered_time['LocationLat'] >= lat_range[0]) & 
        (chunk_filtered_time['LocationLat'] <= lat_range[1]) &
        (chunk_filtered_time['LocationLng'] >= long_range[0]) & 
        (chunk_filtered_time['LocationLng'] <= long_range[1])
    ]

    if not chunk_filtered.empty:
        filtered_chunks.append(chunk_filtered)

# Concatenate all filtered chunks into a single DataFrame
filtered_df = pd.concat(filtered_chunks)

# Convert 'StartTime(UTC)' and 'EndTime(UTC)' to US/Pacific and add as new columns
filtered_df['StartTime(Local)'] = filtered_df['StartTime(UTC)'].apply(lambda x: utc_tz.localize(datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).astimezone(pacific_tz).strftime('%Y-%m-%d %H:%M:%S'))
filtered_df['EndTime(Local)'] = filtered_df['EndTime(UTC)'].apply(lambda x: utc_tz.localize(datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).astimezone(pacific_tz).strftime('%Y-%m-%d %H:%M:%S'))

# %%
filtered_df.to_csv('TrafficEvents_Sep1_Nov30_Sacramento.csv', index=False)

# %%
filtered_df.count()

# %%
!ls -lah

# %% [markdown]
# # Road point processing

# %%
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import folium
from folium.plugins import FastMarkerCluster

# %%
traffic_events_df = pd.read_csv('TrafficEvents_Sep1_Nov30_Sacramento.csv')
traffic_events_gdf = gpd.GeoDataFrame(
    traffic_events_df,
    geometry=gpd.points_from_xy(traffic_events_df.LocationLng, traffic_events_df.LocationLat)
)
traffic_events_gdf.crs = "EPSG:4326"

# %%
traffic_events_df.head()

# %%
traffic_events_df.iloc[0]

# %%
sensors_df = pd.read_csv('PEMS03_attr.csv')
sensors_df.head()
sensors_gdf = gpd.GeoDataFrame(
    sensors_df,
    geometry=gpd.points_from_xy(sensors_df.Longitude, sensors_df.Latitude)
)
sensors_gdf.crs = "EPSG:4326"

# %%
sensors_df

# %%
print(traffic_events_df['Type'].unique())

# %%
print("Unique streets in traffic events data:")
print(traffic_events_df['Street'].unique())

# %%
print(sensors_df['Fwy'].unique())

# %% [markdown]
# ## All data points: Traffic v.s. Sensor

# %%
def add_circle_markers(data, map_object, color, popup_column=None):
    for idx, row in data.iterrows():
        popup_text = row[popup_column] if popup_column else ''
        folium.CircleMarker(
            location=[row['geometry'].y, row['geometry'].x],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(map_object)

# %%
m = folium.Map(location=[38.575764, -121.478851], zoom_start=10)

add_circle_markers(traffic_events_gdf, m, 'red', popup_column='Street')
add_circle_markers(sensors_gdf, m, 'blue', popup_column='ID')

m

# %% [markdown]
# ## Non highway (filter with street name)

# %%
non_highway_pattern = r'\b(?:St|Street|Blvd|Boulevard|Rd|Road|Dr|Drive|Ave|Avenue|Ln|Lane|Way|Alley|Aly|Walk|Cir|Ct|Pkwy|Trl|Pl|Xing|Expy)\b'

traffic_events_non_highway = traffic_events_df[traffic_events_df['Street'].str.contains(non_highway_pattern, case=False, regex=True)]

traffic_events_non_highway_gdf = gpd.GeoDataFrame(
    traffic_events_non_highway,
    geometry=gpd.points_from_xy(traffic_events_non_highway.LocationLng, traffic_events_non_highway.LocationLat),
    crs="EPSG:4326"
)

m = folium.Map(location=[38.575764, -121.478851], zoom_start=10)


add_circle_markers(traffic_events_non_highway_gdf, m, 'red', popup_column='Street')
add_circle_markers(sensors_gdf, m, 'blue', popup_column='ID')

# Display the map
m

# %%
non_highway_pattern = r'\b(?:St|Street|Blvd|Boulevard|Rd|Road|Dr|Drive|Ave|Avenue|Ln|Lane|Way|Alley|Aly|Walk|Cir|Ct|Pkwy|Trl|Pl|Xing|Expy|Broadway|Mall)\b'

traffic_events_non_highway = traffic_events_df[~traffic_events_df['Street'].str.contains(non_highway_pattern, case=False, regex=True)]

traffic_events_non_highway_gdf = gpd.GeoDataFrame(
    traffic_events_non_highway,
    geometry=gpd.points_from_xy(traffic_events_non_highway.LocationLng, traffic_events_non_highway.LocationLat),
    crs="EPSG:4326"
)

m = folium.Map(location=[38.575764, -121.478851], zoom_start=10)


add_circle_markers(traffic_events_non_highway_gdf, m, 'red', popup_column='Street')
add_circle_markers(sensors_gdf, m, 'blue', popup_column='Name')

m

# %%
highway_streets = traffic_events_df[~traffic_events_df['Street'].str.contains(non_highway_pattern, case=False, regex=True)]
unique_highway_streets = highway_streets['Street'].unique()
print(unique_highway_streets)

# %%
# verify those are not realted to sensor
highway_features = [
    'Vietnam Veterans Memorial Bridge', # highway
    'Garden Hwy',
    'Pedestrian Overcrossing', # highway
    'Orchard Loop',
    'Tower Bridge Gtwy',
    'State Highway 113',
    'Caltrans Maintenance Worker Memorial Bridge', # highway
    'Arden-Garden Connector'
]

filtered_highway_events = traffic_events_df[traffic_events_df['Street'].isin(highway_features)]

filtered_highway_events_gdf = gpd.GeoDataFrame(
    filtered_highway_events,
    geometry=gpd.points_from_xy(filtered_highway_events.LocationLng, filtered_highway_events.LocationLat),
    crs="EPSG:4326"
)

m = folium.Map(location=[38.575764, -121.478851], zoom_start=10)

add_circle_markers(filtered_highway_events_gdf, m, 'red', popup_column='Street')
add_circle_markers(sensors_gdf, m, 'blue', popup_column='ID')

m

# %%
# Define the list of highway candidates
highway_candidates = [
    'CA-16 E', 'I-80 W', 'CA-99 N', 'I-5 N', 'El Dorado Fwy W', 'CA-65 N',
    'Capital City Fwy W', 'CA-99 S', 'US-50 E', 'CA-65 S', 'Capital City Fwy E',
    'I-80 E', 'S Sacramento Fwy', 'I-5 S', 'CA-160 N', 'CA-16 W', 'I-80 Bus E',
    'CA-160 S', 'US-50 W', 'I-80 Bus W', 'El Dorado Fwy E', 'Vietnam Veterans Memorial Bridge',
    'Pedestrian Overcrossing', 'Caltrans Maintenance Worker Memorial Bridge'
]

highway_events_df = traffic_events_df[traffic_events_df['Street'].isin(highway_candidates)]
non_highway_events_df = traffic_events_df[~traffic_events_df['Street'].isin(highway_candidates)]

# %%
map_highway = folium.Map(location=[38.575764, -121.478851], zoom_start=10)
map_non_highway = folium.Map(location=[38.575764, -121.478851], zoom_start=10)

# Convert filtered events to GeoDataFrames
gdf_highway = gpd.GeoDataFrame(highway_events_df, geometry=gpd.points_from_xy(highway_events_df.LocationLng, highway_events_df.LocationLat))
gdf_non_highway = gpd.GeoDataFrame(non_highway_events_df, geometry=gpd.points_from_xy(non_highway_events_df.LocationLng, non_highway_events_df.LocationLat))

# Add highway-related traffic events and sensor points to the first map
add_circle_markers(gdf_highway, map_highway, 'red', popup_column='Street')
add_circle_markers(sensors_gdf, map_highway, 'blue', popup_column='ID')

# Add non-highway traffic events and sensor points to the second map
add_circle_markers(gdf_non_highway, map_non_highway, 'green', popup_column='Street')
add_circle_markers(sensors_gdf, map_non_highway, 'blue', popup_column='ID')

map_highway.save('highway_and_sensors_map.html')
map_non_highway.save('non_highway_and_sensors_map.html')

# %%
map_highway

# %%
highway_events_df.head()

# %%
gdf_highway_events = gpd.GeoDataFrame(highway_events_df, geometry=gpd.points_from_xy(highway_events_df.LocationLng, highway_events_df.LocationLat))
gdf_sensors = gpd.GeoDataFrame(sensors_df, geometry=gpd.points_from_xy(sensors_df.Longitude, sensors_df.Latitude))

# gdf_highway_events.crs = "EPSG:4326"
# gdf_sensors.crs = "EPSG:4326"
gdf_highway_events.crs = {'init': "epsg:4326"}
gdf_sensors.crs = {'init': "epsg:4326"}

# gdf_highway_events = gdf_highway_events.to_crs("EPSG:2226")  # NAD83 / California zone 2 (ftUS)
# gdf_sensors = gdf_sensors.to_crs("EPSG:2226")
gdf_highway_events = gdf_highway_events.to_crs(epsg=2226)  # NAD83 / California zone 2 (ftUS)
gdf_sensors = gdf_sensors.to_crs(epsg=2226)

# %% [markdown]
# ## Highway events matching sensors

# %%
print(sensors_df['Fwy'].unique())

# %%
# Define the list of highway candidates
# Fwy 51 = Capital City Fwy
# CA-16 overlaps 5 & 50, El Dorado overlaps 50
highway_candidates = [
    'US-50 E', 'I-80 W', 'I-5 N', 'CA-99 N', 'CA-65 N', 'I-80 Bus E', 'Capital City Fwy E', 'CA-16 E', 'El Dorado Fwy E',
    'US-50 W', 'I-80 E', 'I-5 S', 'CA-99 S', 'CA-65 S', 'I-80 Bus W', 'Capital City Fwy W', 'CA-16 W', 'El Dorado Fwy W', 'S Sacramento Fwy' # overlap 99
]

highway_events_df = traffic_events_df[traffic_events_df['Street'].isin(highway_candidates)]

# %% [markdown]
# # Define the list of highway candidates
# highway_candidates = [
#     # 'CA-160 N', 'CA-160 S', 'Vietnam Veterans Memorial Bridge', 'Pedestrian Overcrossing', 'Caltrans Maintenance Worker Memorial Bridge'
# ]
# 
# highway_events_df = traffic_events_df[traffic_events_df['Street'].isin(highway_candidates)]

# %%
print(highway_events_df.shape)

# %%
highway_events_df.head()

# %%
map_highway = folium.Map(location=[38.575764, -121.478851], zoom_start=10)

# Convert filtered events to GeoDataFrames
gdf_highway = gpd.GeoDataFrame(highway_events_df, geometry=gpd.points_from_xy(highway_events_df.LocationLng, highway_events_df.LocationLat))

# Add highway-related traffic events and sensor points to the first map
add_circle_markers(gdf_highway, map_highway, 'red', popup_column='Street')
add_circle_markers(sensors_gdf, map_highway, 'blue', popup_column='Fwy')

map_highway

# %%
# Have to reprojection to calculate distance!
gdf_highway_events = gpd.GeoDataFrame(highway_events_df, geometry=gpd.points_from_xy(highway_events_df.LocationLng, highway_events_df.LocationLat))
gdf_sensors = gpd.GeoDataFrame(sensors_df, geometry=gpd.points_from_xy(sensors_df.Longitude, sensors_df.Latitude))

gdf_highway_events.crs = 'EPSG:4326'
gdf_sensors.crs = 'EPSG:4326'
# gdf_highway_events = gdf_highway_events.set_crs(4326, allow_override=True)
# gdf_sensors = gdf_sensors.set_crs(4326, allow_override=True)

print(gdf_highway_events.crs)

# gdf_highway_events = gdf_highway_events.to_crs("EPSG:2226")  # NAD83 / California zone 2 (ftUS)
# gdf_sensors = gdf_sensors.to_crs("EPSG:2226")
gdf_highway_events = gdf_highway_events.to_crs('EPSG:2226')  # NAD83 / California zone 2 (ftUS)
gdf_sensors = gdf_sensors.to_crs('EPSG:2226')

print(gdf_highway_events.crs)

# %%
sensor100_event_df, opp_match_df = analyze_sensor_event_relations(100)
print(sensor100_event_df)
print(opp_match_df)

# %%
# considering road direction
def analyze_sensor_event_relations(buffer_size): 
    buffer_size_ft = 3.2808333333*buffer_size    # meter to ftUS
    gdf_sensors[f'buffer_{buffer_size}'] = gdf_sensors.geometry.buffer(buffer_size_ft)
    
    joined = gpd.sjoin(gdf_highway_events, gdf_sensors.set_geometry(f'buffer_{buffer_size}'), how='inner', op='within')
    events = joined.groupby('ID')['EventId'].apply(lambda x: ','.join(x)).reset_index(name=f'EventId_{buffer_size}')
    
    sensors_with_events = sensors_df.merge(events, on='ID', how='left')
    # print(f"Number of no traffic event sensors: {sensors_with_events[f'EventId_{buffer_size}'].isnull().sum()}")
    
    # For each event ID, merge event details
    expanded_rows = []
    for idx, row in sensors_with_events.dropna(subset=[f'EventId_{buffer_size}']).iterrows():
        event_ids = row[f'EventId_{buffer_size}'].split(',')
        for event_id in event_ids:
            event_details = highway_events_df[highway_events_df['EventId'] == event_id]
            for _, event_row in event_details.iterrows():
                combined_row = {**row.to_dict(), **event_row.to_dict()}
                expanded_rows.append(combined_row)
    
    expanded_df = pd.DataFrame(expanded_rows)
    columns_to_drop = ['Unnamed: 0', f'EventId_{buffer_size}', 'StartTime(UTC)', 'EndTime(UTC)', 'TimeZone', 'Number']
    expanded_df = expanded_df.drop(columns=columns_to_drop)
    
    # filter out rows {sensor, event} not in same direction
    oppo_match_rows = []
    for idx, row in expanded_df.iterrows():
        if row['Street'] == 'S Sacramento Fwy':
            if row['Dir'] != 'S':
                oppo_match_rows.append(row)
                expanded_df.drop(index=idx, inplace=True)
        else:
            if row['Dir'] != row['Street'][-1]:
                oppo_match_rows.append(row)
                expanded_df.drop(index=idx, inplace=True)
    oppo_match_df = pd.DataFrame(oppo_match_rows)
    
    expanded_df.to_csv(f'Sensor_{buffer_size}m_dir_Events_Sep1_Nov30_Sacramento.csv', index=False)
    oppo_match_df.to_csv(f'Sensor_{buffer_size}m_opp_Events_Sep1_Nov30_Sacramento.csv', index=False)

    return expanded_df, oppo_match_df

# %%


# %%
# without considering road direction
def analyze_sensor_event_relations(buffer_size): 
    buffer_size_ft = 3.2808333333*buffer_size    # meter to ftUS
    gdf_sensors[f'buffer_{buffer_size}'] = gdf_sensors.geometry.buffer(buffer_size_ft)
    
    joined = gpd.sjoin(gdf_highway_events, gdf_sensors.set_geometry(f'buffer_{buffer_size}'), how='inner', op='within')
    events = joined.groupby('ID')['EventId'].apply(lambda x: ','.join(x)).reset_index(name=f'EventId_{buffer_size}')

    sensors_with_events = sensors_df.merge(events, on='ID', how='left')
    print(f"Number of no traffic event sensors: {sensors_with_events[f'EventId_{buffer_size}'].isnull().sum()}")
    
    # For each event ID, merge event details
    expanded_rows = []
    for idx, row in sensors_with_events.dropna(subset=[f'EventId_{buffer_size}']).iterrows():
        event_ids = row[f'EventId_{buffer_size}'].split(',')
        for event_id in event_ids:
            event_details = highway_events_df[highway_events_df['EventId'] == event_id]
            for _, event_row in event_details.iterrows():
                combined_row = {**row.to_dict(), **event_row.to_dict()}
                expanded_rows.append(combined_row)
    
    expanded_df = pd.DataFrame(expanded_rows)
    columns_to_drop = ['Unnamed: 0', f'EventId_{buffer_size}', 'StartTime(UTC)', 'EndTime(UTC)', 'TimeZone', 'Number']
    expanded_df = expanded_df.drop(columns=columns_to_drop)
    
    expanded_df.to_csv(f'Sensor_{buffer_size}m_Events_Sep1_Nov30_Sacramento.csv', index=False)

    return expanded_df

# %%
sensor100_event_df = analyze_sensor_event_relations(100)
sensor100_event_df

# %%
sensor50_event_df = analyze_sensor_event_relations(50)

# %%
sensor50_event_df

# %%
sensor200_event_df = analyze_sensor_event_relations(200)

# %%
sensor200_event_df.columns

# %%
print(sensor200_event_df['Type'].unique())

# %%



