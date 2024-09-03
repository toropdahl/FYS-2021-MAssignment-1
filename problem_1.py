import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def csv_to_test_and_train():
    data = pd.read_csv('SpotifyFeatures.csv')  
    num_entries, num_features = data.shape

    print(f"Number of entries: {num_entries}")
    print(f"Number of features: {num_features}")

    data = data[['liveness', 'loudness', 'genre']]
    data = data[data['genre'].isin(['Pop', 'Classical'])]

    data['label'] = data['genre'].apply(lambda x: 1 if x == 'Pop' else 0)
    class_counts = data['label'].value_counts()
    print(f"Number of 'Pop' samples: {class_counts[1]}")
    print(f"Number of 'Classical' samples: {class_counts[0]}")

    X = data[['liveness', 'loudness']].to_numpy()  # Feature matrix
    y = data['label'].to_numpy()  # Target vector


    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = csv_to_test_and_train()
    print(f"Training set size: {len(y_train)} samples, Test set size: {len(y_test)} samples")
    print(f"Training set distribution: Pop = {np.sum(y_train)}, Classical = {len(y_train) - np.sum(y_train)}")
    print(f"Test set distribution: Pop = {np.sum(y_test)}, Classical = {len(y_test) - np.sum(y_test)}")
