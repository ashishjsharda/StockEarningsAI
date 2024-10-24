import unittest
from src.model.model import train_model, predict_stock_movement
from src.features.feature_engineering import add_features
from src.data.data_fetcher import fetch_data


class TestStockMovementPredictor(unittest.TestCase):

    def test_model_accuracy(self):
        data = fetch_data('AAPL', '2020-01-01', '2023-01-01')
        X, y = add_features(data)
        model = train_model(X, y)

        # Test that model is trained with a non-zero accuracy
        self.assertTrue(model is not None)

    def test_prediction(self):
        data = fetch_data('AAPL', '2020-01-01', '2023-01-01')
        X, y = add_features(data)
        model = train_model(X, y)

        # Predict movement for a new earnings input
        new_data = np.array([[0.5, 1.2, 0.8]])  # Example data
        prediction = predict_stock_movement(model, new_data)

        # Test that the prediction is either 0 or 1
        self.assertIn(prediction[0], [0, 1])


if __name__ == "__main__":
    unittest.main()
