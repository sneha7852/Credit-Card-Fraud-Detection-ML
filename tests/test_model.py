import unittest
from src.data_preprocessing import load_data, preprocess_data
from src.model import build_models

class TestFraudDetection(unittest.TestCase):

    def setUp(self):
        self.data = load_data('data/sample_data.csv')
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_data(self.data)
        self.models = build_models()

    def test_data_preprocessing(self):
        self.assertEqual(self.X_train.shape[0], 0.7 * self.data.shape[0])
        self.assertEqual(self.X_test.shape[0], 0.3 * self.data.shape[0])
    
    def test_model_training(self):
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            self.assertEqual(len(y_pred), len(self.y_test))

if __name__ == '__main__':
    unittest.main()
