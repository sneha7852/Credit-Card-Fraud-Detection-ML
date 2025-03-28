import pickle
from sklearn.metrics import classification_report
from src.data_preprocessing import load_data, preprocess_data

def evaluate_model(model_filepath, X_test, y_test):
    """Load the model and evaluate it on the test data."""
    with pickle.load(open(model_filepath, 'rb')) as model:
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    data = load_data('data/sample_data.csv')
    _, X_test, _, y_test = preprocess_data(data)
    evaluate_model('models/random_forest.pkl', X_test, y_test)
