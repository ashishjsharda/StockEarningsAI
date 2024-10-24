from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model(X, y):
    """
    Train a RandomForest model on the stock data.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return model


def predict_stock_movement(model, new_data):
    """
    Use the trained model to predict stock movement (1 = UP, 0 = DOWN).
    """
    prediction = model.predict(new_data)
    return prediction
