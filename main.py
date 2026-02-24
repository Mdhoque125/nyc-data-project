import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# NYC Open Data API endpoint (example dataset)
DATA_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

def fetch_data(limit=1000):
    params = {
        "$limit": limit,
        "$where": "created_date >= '2024-01-01T00:00:00'"
    }
    
    response = requests.get(DATA_URL, params=params)
    data = response.json()
    return pd.DataFrame(data)

def clean_data(df):
    df['created_date'] = pd.to_datetime(df['created_date'])
    df['hour'] = df['created_date'].dt.hour
    return df

def analyze_by_hour(df):
    hourly_counts = df.groupby('hour').size()
    return hourly_counts

def plot_results(hourly_counts):
    plt.figure()
    hourly_counts.plot(kind='line')
    plt.title("NYC Transit Complaints by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Complaints")
    plt.show()

if __name__ == "__main__":
    print("Fetching NYC subway-related data...")
    
    df = fetch_data()
    df = clean_data(df)
    hourly_counts = analyze_by_hour(df)
    
    print("Plotting results...")
    plot_results(hourly_counts)
    from sklearn.linear_model import LinearRegression
import numpy as np

def train_model(df):
    hourly = df.groupby('hour').size().reset_index()
    X = hourly[['hour']]
    y = hourly[0]
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model
def predict_delay(model, hour):
    prediction = model.predict([[hour]])
    print(f"Estimated complaint volume at {hour}:00 = {int(prediction[0])}")