"""
preprocessing.py
-----------------
Shared cleaning + feature-engineering pipeline, extracted from
UberRideDataAnalysis.ipynb so every topic notebook can reuse it
instead of repeating the code.
"""

import numpy as np
import pandas as pd


def load_and_clean_data(raw_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path)

    # --- Duplicates ------------------------------------------------------
    df.drop_duplicates(inplace=True)

    # --- Missing values: reason columns -> explicit label -----------------
    reason_columns = [
        'Incomplete Rides Reason',
        'Reason for cancelling by Customer',
        'Driver Cancellation Reason',
    ]
    for col in reason_columns:
        df[col] = df[col].fillna('Not Applicable')

    # --- Missing values: numeric columns -> coerce + median fill -----------
    numeric_columns = [
        'waiting time', 'Avg CTAT', 'Booking Value',
        'Ride Distance', 'Driver Ratings', 'Customer Rating',
    ]
    for num in numeric_columns:
        coerced = pd.to_numeric(df[num], errors='coerce')
        n_newly_null = coerced.isnull().sum() - df[num].isnull().sum()
        if n_newly_null > 0:
            print(f"Note: {n_newly_null} non-numeric values in '{num}' were coerced to NaN.")
        df[num] = coerced.fillna(coerced.median())

    # --- Missing values: categorical columns -> mode fill -------------------
    categorical_cols = ['Vehicle Type', 'Pickup Location', 'Drop Location',
                         'Payment Method', 'Customer ID']
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # --- Date features -----------------------------------------------------
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Date_only'] = df['Date'].dt.date
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.month_name()
    df['Day'] = df['Date'].dt.day
    df['Day_of_week'] = df['Date'].dt.day_name()
    df['Is_Weekend'] = np.where(df['Date'].dt.dayofweek >= 5, 'Yes', 'No')

    # --- Time features -------------------------------------------------------
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')
    df['Hour'] = df['Time'].dt.hour

    def day_night(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    df['Day-Night'] = df['Hour'].apply(day_night)

    def peak_hour(hour):
        return 'Peak Hour' if (7 <= hour <= 10 or 17 <= hour <= 20) else 'Non-Peak Hour'

    df['Peak_Hour'] = df['Hour'].apply(peak_hour)

    return df


if __name__ == "__main__":
    import os

    cleaned = load_and_clean_data("../data/Uber_Drive_Raw.csv")
    os.makedirs("../data/processed", exist_ok=True)
    cleaned.to_csv("../data/processed/uber_cleaned.csv", index=False)
    print(f"Cleaned data saved: {cleaned.shape[0]} rows, {cleaned.shape[1]} columns")
