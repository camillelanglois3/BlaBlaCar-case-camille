import holidays
import pandas as pd


def is_holiday(row):
    """
    Determines whether a given date is a public holiday in the driver's signup country.

    Args:
        row (pd.Series): A row from a DataFrame that must contain the following columns:
            - 'fixed_signup_country': ISO 3166-1 alpha-2 country code (e.g., 'FR', 'DE').
            - 'segment_datetime': A datetime object representing the departure date and time.

    Returns:
        bool: True if the date is a public holiday in the specified country, False otherwise.
    """
    country = row['fixed_signup_country']
    date = row['segment_datetime'].date()
    try:
        country_holidays = holidays.CountryHoliday(country)
        return date in country_holidays
    except:
        return False
    

def feature_engineering():
    """
        Performs feature engineering on the input dataset.        
    """
    print("------ Feature engineering ------")
    df = pd.read_parquet('../data/processed/clean_dataset.parquet')

    # Simple features creation
    df["hours_before_departure"] = ((df["segment_datetime"] - df["published_date"]).dt.total_seconds() / 3600).round(2)
    df["driver_account_age_days"] = (df["segment_datetime"] - df["signup_date"]).dt.days
    driver_trip_count = df.groupby("driver_id")["trip_id"].nunique().rename("driver_trip_count")
    df = df.merge(driver_trip_count, on="driver_id", how="left")

    # Temporal features creation
    df["departure_hour"] = df["segment_datetime"].dt.hour
    df["departure_weekday"] = df["segment_datetime"].dt.weekday # 0 = monday
    df["is_weekend"] = df["departure_weekday"].isin([5, 6]).astype(int)
    df['fixed_signup_country'] = df['fixed_signup_country'].str.strip()
    df['is_holiday'] = df.apply(is_holiday, axis=1)

    # Geographical features creation
    df["from_cluster"] = df["from_lat"].round(1).astype(str) + "," + df["from_lon"].round(1).astype(str)
    df["to_cluster"] = df["to_lat"].round(1).astype(str) + "," + df["to_lon"].round(1).astype(str)

    df["segment_cluster"] = df["from_cluster"] + " -> " + df["to_cluster"]

    df["segment_cluster_popularity"] = df.groupby("segment_cluster")["segment_id"].transform("count")
    df["from_cluster_popularity"] = df.groupby("from_cluster")["segment_id"].transform("count")
    df["to_cluster_popularity"] = df.groupby("to_cluster")["segment_id"].transform("count")
    df.drop(['from_cluster', 'to_cluster', 'segment_cluster'], axis=1, inplace=True)

    # Interaction features creation
    df["price_per_km"] = df["unit_seat_price_eur"] / df["segment_distance_km"]

    df["is_long_trip"] = (df["segment_distance_km"] > 500).astype(int)

    df["price_x_popularity"] = df["unit_seat_price_eur"] * df["segment_cluster_popularity"]
    df["seats_x_distance"] = df["seat_offered_count"] * df["segment_distance_km"]

    print("File saved in data/processed/enriched_dataset.parquet")
    df.to_parquet("../data/processed/enriched_dataset_2.parquet")
