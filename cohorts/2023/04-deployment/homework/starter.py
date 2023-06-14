#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
import os
import sys


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

year = 2022
month = 4

def read_data(filename):
 
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    return df


df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


df['y_pred'] = y_pred
df_result = df[['ride_id', 'y_pred']].copy()

# Mean predicted duration
mean_duration = np.mean(y_pred)
print(f'The mean predicted duration of yellow taxi trips in April 2022 is {mean_duration}')

# Create the 'data' directory if it doesn't exist
output_dir ='data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the output file path
output_file = os.path.join(output_dir, f'result_{year}_{month}.parquet')

# Save the DataFrame to the Parquet file
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

def run():
    year = int(sys.argv[1]) # 2022
    month = int(sys.argv[2]) # 4

if __name__ == '__main__':
    run()