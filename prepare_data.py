import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler , MinMaxScaler

def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    earth_radius_km = 6371  # Earth's radius in kilometers
    distance = earth_radius_km * c

    return distance


def prepare_data(df):
    
    df.drop(columns='id', axis=1, inplace=True)
    df.drop(columns="dropoff_datetime", axis=1,inplace=True)
    
    
    df['distance'] = df.apply(lambda row: haversine_distance(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis=1)
    
    
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['month'] = df['pickup_datetime'].dt.month
    df['week'] = df['pickup_datetime'].dt.weekday
    df['day'] = df['pickup_datetime'].dt.day
    df['hour'] = df['pickup_datetime'].dt.hour
    df['weekday'] = df['pickup_datetime'].dt.day_name()
    df['time_of_day'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    
    
    df['trip_duration'] = np.log1p(df['trip_duration'])
    
    
    
    df['euclidean_distance'] = ((df['pickup_longitude'] - df['dropoff_longitude'])**2 +
                            (df['pickup_latitude'] - df['dropoff_latitude'])**2)**0.5
    df['manhattan_distance'] = abs(df['pickup_longitude'] - df['dropoff_longitude']) + \
                            abs(df['pickup_latitude'] - df['dropoff_latitude'])
                            
    df['distance'] = np.log1p(df['distance'])
    df['euclidean_distance'] = np.log1p(df['euclidean_distance'])
    df['manhattan_distance'] = np.log1p(df['manhattan_distance'])
    
    
    df = pd.get_dummies(df,columns=['vendor_id','store_and_fwd_flag','weekday','time_of_day'])
    
    
    df.drop(columns='pickup_datetime', axis=1, inplace=True)
    
    return df