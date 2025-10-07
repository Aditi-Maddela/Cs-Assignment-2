import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.dropna()
    
    if df['Result'].nunique() == 2:
        if set(df['Result'].unique()) == {-1, 1}:
            df['Result'] = df['Result'].map({-1: 0, 1: 1})

    X = df.drop('Result', axis=1)
    y = df['Result']

    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes

    return X, y

def build_and_train(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
    ensemble.fit(X_train, y_train)
    return ensemble

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def main(args):
    df = load_data(args.data)
    X, y = preprocess(df)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    model = build_and_train(X_train, y_train)

    joblib.dump(model, 'phishing_ensemble_model.joblib')

    evaluate(model, X_test, y_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to phishing CSV file')
    args = parser.parse_args()
    main(args)
