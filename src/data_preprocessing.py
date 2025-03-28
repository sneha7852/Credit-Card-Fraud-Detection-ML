import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the dataset: scale features and split into train/test sets."""
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    data = load_data('data/sample_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
