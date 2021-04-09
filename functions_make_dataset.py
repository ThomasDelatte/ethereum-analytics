import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ="/home/tdelatte/new-projects/ethereum-analytics/key/ethereum-analytics-309308-6c01508bc0b8.json"

import pandas as pd
import pickle

def load_data_from_bigquery(QUERY):
    """Loads data from BigQuery through GCP."""
    query_job = client.query(QUERY) # API request
    df = query_job.to_dataframe()
    return df

def add_mined_blocks(df):
    for i, row in df.iterrows():
        eth_address = row.ethereum_address
        if i % 500 == 0:
            print(f"We are at the {i}th row!")
        try:
            mined = len(eth.get_mined_blocks_by_address(address=eth_address))
        except:
            continue
        if mined:
            df.iat[i, 4] = mined
    return df