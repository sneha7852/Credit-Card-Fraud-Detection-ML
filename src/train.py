from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.data_preprocessing import load_data, preprocess_data
from src.model import build_models

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = build_models()
    results = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[model_name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred)
        }
    
    return results

if __name__ == "__main__":
    data = load_data('data/sample_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    print(results)
