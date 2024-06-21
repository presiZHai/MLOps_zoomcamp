#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import pandas as pd
import os
import boto3
from botocore.exceptions import NoCredentialsError

# Function to calculate mean predicted duration
def calculate_mean_predicted_duration(df):
    return df['predicted_duration'].mean()

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process predicted durations for a specific month and year.')
    parser.add_argument('year', type=int, help='Year (e.g., 2023)')
    parser.add_argument('month', type=int, help='Month as a number (e.g., 5 for May)')
    return parser.parse_args()

# Function to upload file to S3
def upload_to_s3(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = os.path.basename(file_name)

    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_name, bucket, object_name)
        print(f"File {file_name} uploaded to {bucket}/{object_name}")
    except NoCredentialsError:
        print("Credentials not available")

# Main execution
if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()

    # Load model
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    # Function to read data
    def read_data(filename):
        df = pd.read_parquet(filename)
        
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        
        return df

    # Read the data for the specified month and year
    file_url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year:04d}-{args.month:02d}.parquet'
    df = read_data(file_url)

    # Make predictions
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # Calculate and print the standard deviation of predictions
    std_dev = y_pred.std()
    print("Standard Deviation of y_pred:", std_dev)

    # Add ride_id and predicted_duration to the dataframe
    df['ride_id'] = f'{args.year:04d}/{args.month:02d}_' + df.index.astype('str')
    df['predicted_duration'] = y_pred 

    # Create a dataframe with the results
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })

    # Save the results as a Parquet file
    output_file = 'results.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    # Print the size of the output file
    file_size = os.path.getsize(output_file)
    print(f"Size of the output file '{output_file}': {file_size} bytes")

    # Calculate and print the mean predicted duration
    mean_predicted_duration = calculate_mean_predicted_duration(df)
    print(f"Mean predicted duration for {args.year}/{args.month}: {mean_predicted_duration}")

    # Upload the Parquet file to S3
    bucket_name = 'mlflow-models-waley'  # Replace with your bucket name
    upload_to_s3(output_file, bucket_name)