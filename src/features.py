import numpy as np
import pandas as pd


def make_incident_labels(values, threshold):
    """
    Incident = 1 when current metric exceeds threshold.
    """
    return (values >= threshold).astype(int)


def make_supervised_dataset(values, incident_labels, window_size=20, horizon=5):
    """
    X_t = previous W values
    y_t = 1 if any incident occurs in the next H steps
    """
    X, y = [], []
    n = len(values)

    for t in range(window_size, n - horizon):
        past_window = values[t - window_size:t]
        future_incident = incident_labels[t:t + horizon].max()

        # simple raw-window features
        features = list(past_window)

        # add summary features
        features.extend([
            np.mean(past_window),
            np.std(past_window),
            np.min(past_window),
            np.max(past_window),
            past_window[-1] - past_window[0],   # slope proxy
            past_window[-1],                    # latest value
        ])

        X.append(features)
        y.append(future_incident)

    return np.array(X), np.array(y)