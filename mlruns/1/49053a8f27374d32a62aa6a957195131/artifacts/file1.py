import sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_recall_curve, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


import mlflow
import mlflow.sklearn

# mlflow.delete_experiment(2)


import os
# os.makedirs('mlruns', exist_ok=True)

# mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("exp1-4")

df = load_wine()

X = df.data
y = df.target


X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,
    random_state=42,
    test_size=0.10
)

n_estinators = 5
max_depth = 5


# starting mlflow code
# tracking each run using context manager of mlflow
with mlflow.start_run():
    rf = RandomForestClassifier(
        n_estimators=n_estinators,
        max_depth=max_depth,
        random_state=42
    )

    rf.fit(
        X_train, y_train
    )

    y_preds = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_preds)

    # logging metrics
    mlflow.log_metric('accuracy', accuracy)

    # logging hyperparams inside mlflow for tracking
    mlflow.log_param("n_estimators", n_estinators)
    mlflow.log_param("max_depth", max_depth)

    mlflow.log_params(
        {
            "n_estimators":n_estinators,
            "max_depth": max_depth
        }
    )

    cm = confusion_matrix(
        y_test, y_preds
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=df.target_names,
        yticklabels=df.target_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("Confusion Matrix.png")
    plt.close()

    mlflow.log_artifact(os.path.abspath("Confusion Matrix.png"))
    mlflow.log_artifact("src/file1.py")


    print(f"Accuracy: {accuracy}")
    print(cm)
