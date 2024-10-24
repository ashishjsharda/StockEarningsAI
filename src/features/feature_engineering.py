import numpy as np


def add_features(data):
    """
    Feature engineering for stock data and earnings.
    """
    # Calculate earnings surprise
    data['Earnings_Surprise'] = data['EPSActual'] - data['EPSEstimate']

    # Shift stock price returns to align with the next day movement
    data['Next_Day_Return'] = data['Returns'].shift(-1)

    # Create a target variable: 1 for up, 0 for down
    data['Stock_Movement'] = np.where(data['Next_Day_Return'] > 0, 1, 0)

    # Drop NaN values
    data = data.dropna()

    # Select features and target
    X = data[['Earnings_Surprise', 'EPSActual', 'EPSEstimate']]
    y = data['Stock_Movement']

    return X, y
