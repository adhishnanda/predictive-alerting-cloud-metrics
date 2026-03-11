import os
import requests
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/cpu_utilization_asg_misconfiguration.csv"


def download_dataset(save_path="data/cpu_utilization_asg_misconfiguration.csv"):
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(save_path):
        r = requests.get(DATA_URL, timeout=30)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
    return save_path


def load_dataset(path="data/cpu_utilization_asg_misconfiguration.csv"):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df