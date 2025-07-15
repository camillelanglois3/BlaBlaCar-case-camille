import pandas as pd
from geopy.distance import geodesic

def calc_geo_distance(row):
    """
    Computes the geographical distance (geodesic distance) in kilometers between two coordinates.

    Args:
        row (pd.Series): A row from a DataFrame containing the following columns:
            - 'from_lat' (float): Latitude of the departure point.
            - 'from_lon' (float): Longitude of the departure point.
            - 'to_lat' (float): Latitude of the arrival point.
            - 'to_lon' (float): Longitude of the arrival point.

    Returns:
        float: The distance in kilometers between the two points.
    """
    return geodesic((row['from_lat'], row['from_lon']), (row['to_lat'], row['to_lon'])).km

def check_coord_ranges(df, min_lat=-90, max_lat=90, min_lon=-180, max_lon=180):
    """
    Checks whether geographic coordinates in the DataFrame fall within valid latitude and longitude ranges.

    Args:
        df (pd.DataFrame): A DataFrame containing the following columns:
            - 'from_lat': Latitude of the departure point.
            - 'from_lon': Longitude of the departure point.
            - 'to_lat': Latitude of the arrival point.
            - 'to_lon': Longitude of the arrival point.
        min_lat (float, optional): Minimum valid latitude. Defaults to -90.
        max_lat (float, optional): Maximum valid latitude. Defaults to 90.
        min_lon (float, optional): Minimum valid longitude. Defaults to -180.
        max_lon (float, optional): Maximum valid longitude. Defaults to 180.

    Returns:
        None. The function prints the number of invalid values for each coordinate column.
    """
    conditions = {
        "from_lat_invalid": ~df["from_lat"].between(min_lat, max_lat),
        "from_lon_invalid": ~df["from_lon"].between(min_lon, max_lon),
        "to_lat_invalid": ~df["to_lat"].between(min_lat, max_lat),
        "to_lon_invalid": ~df["to_lon"].between(min_lon, max_lon),
    }
    for key, cond in conditions.items():
        print(f"{key}: {cond.sum()} invalid values")


def data_preparation(input_file_path):
    """
    Prepares the dataset for modeling by loading and processing the data.

    Args:
        input_file_path (str): Path to the input dataset file.
    """
    print("------ Data preparation ------")
    # Loading dataset
    df = pd.read_csv(input_file_path)

    # Types correction
    # IDs must be integers
    id_cols = ["driver_id", "trip_id", "segment_id", "segment_distance_km", "publication_site_id"]
    for col in id_cols:
        df[col] = df[col].astype(str).str.replace(",", "").astype(int)

    # Dates columns must be in datetime format
    date_cols = ['segment_datetime', 'published_date', 'signup_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Converting the price in float 
    float_cols = ["unit_seat_price_eur"]
    for col in float_cols:
        df[col] = df[col].astype(str).str.replace(",", "").astype(float)

    # 'success' variable definition = if at least 1 seat is confirmed
    df["success"] = df["confirmed_seat_count"] > 0

    # Data cleaning
    # Simple imputation + indicator creation
    df["fixed_signup_country"] = df["fixed_signup_country"].fillna("missing")
    # XX can be considered as not referenced, so treated like a missing value
    df["fixed_signup_country"] = df["fixed_signup_country"].replace({"XX ": "missing"})
    df["fixed_signup_country_grouped"] = df["fixed_signup_country"].apply(lambda x: x if x in ['FR ', 'missing'] else 'foreign')
    df["signup_country_missing"] = df["fixed_signup_country"] == "missing"

    # Inconsistencies suppression
    df = df[df["confirmed_seat_count"] <= df["seat_offered_count"]] 
    df = df[df['segment_distance_km'] > 0]
    df = df[df['published_date'] <= df['segment_datetime']]
    df = df[df["signup_date"] <= df["published_date"]]
    df = df[df['unit_seat_price_eur'] > 0]

    # Correction of the seats count because a lot of lines (1409322) are inconsistent.
    df['seat_left_count_corrected'] = df['seat_offered_count'] - df['confirmed_seat_count']
    df = df[df['seat_left_count_corrected'] >= 0]
    df['seat_left_count'] = df['seat_left_count_corrected']
    df.drop(columns=['seat_left_count_corrected'], inplace=True)

    # Deletion of distance outliers
    df = df[df["segment_distance_km"]<6000]

    # Deletion of lines with 0 seat offered
    df = df[df["seat_offered_count"]>0]

    # Deletion of is_comfort column
    df.drop(['is_comfort'], axis=1, inplace=True)

    print("File saved in data/processed/clean_dataset.parquet")
    df.to_parquet("../data/processed/clean_dataset.parquet")