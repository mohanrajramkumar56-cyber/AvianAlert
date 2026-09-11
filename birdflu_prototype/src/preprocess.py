import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_and_preprocess(path=None):
    if path is None:
        BASE = Path(__file__).resolve().parents[1]
        path = BASE / 'data' / 'synthetic_data.csv'
    df = pd.read_csv(path)
    df = pd.get_dummies(df, columns=['gender','occupation'], drop_first=True)
    X = df.drop(['id','infected'], axis=1)
    y = df['infected']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test
