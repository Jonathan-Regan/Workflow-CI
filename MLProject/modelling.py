import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Aktifkan autolog di luar main agar mencakup seluruh sesi
mlflow.sklearn.autolog()

def main():
    # Set nama eksperimen agar tidak tercampur dengan proyek lain
    mlflow.set_experiment("Heart_Disease_Classification")

    try:
        df = pd.read_csv("heart_preprocessing/heart_preprocessed.csv")
    except FileNotFoundError:
        print("Error: File CSV tidak ditemukan. Pastikan path benar.")
        return

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluasi manual untuk tampilan terminal
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")

if __name__ == "__main__":

    main()

