#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import pandas as pd
import os

# Function to calculate mean predicted duration
def calculate_mean_predicted_duration(df):
    return df['predicted_duration'].mean()

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process predicted durations for a specific month and year.')
    parser.add_argument('year', type=int, help='Year (e.g., 2023)')
    parser.add_argument('month', type=int, help='Month as a number (e.g., 3 for March)')
    return parser.parse_args()

# Main execution
if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()

    # Load model
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    # Function to read and preprocess data
    def read_data(filename):
        df = pd.read_parquet(filename)
        
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df['duration'].dt.total_seconds() / 60
        
        df = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()
        
        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        
        return df

    # Example data URL
    data_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{:04d}-{:02d}.parquet'.format(args.year, args.month)
    
    # Read data
    df = read_data(data_url)

    # Prepare ride_id column
    df['ride_id'] = f'{args.year:04d}/{args.month:02d}_' + df.index.astype('str')

    # Transform categorical features and predict
    dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # Add predicted durations to df_result
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })

    # Save df_result as Parquet
    output_file = 'results.parquet'
    df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)

    # Get the size of the output file
    file_size = os.path.getsize(output_file)
    print(f"Size of the output file '{output_file}': {file_size} bytes")

    # Calculate mean predicted duration
    mean_predicted_duration = calculate_mean_predicted_duration(df_result)

    # Print the mean predicted duration
    print(f"Mean predicted duration for {args.year}/{args.month}: {mean_predicted_duration}")