import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    df = pd.read_csv("./df_clear.csv")
    df = df.set_index("User_ID")
    scaler = StandardScaler()
    to_scale = ['Social_Media_Hours','Exercise_Hours','Sleep_Hours','Screen_Time_Hours','Wearable_Stress_Score']
    df[to_scale] = scaler.fit_transform(df[to_scale])

    X,y = df.drop(columns = ['Academic_Performance']), df['Academic_Performance']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=0.3,
                                                      random_state=42)

    from sklearn.metrics import accuracy_score
    def eval_metrics(actual, pred):
        accuracy = accuracy_score(actual, pred)
        return accuracy

    from sklearn.neighbors import KNeighborsClassifier

    params = {'n_neighbors': [5, 10, 20, 50, 100 ],
          'weights': ['uniform','distance']
     }
    mlflow.set_experiment("mental health knn")
    with mlflow.start_run():

        lr = KNeighborsClassifier()
        clf = GridSearchCV(lr, params, cv = 5)
        clf.fit(X_train, y_train)
        best = clf.best_estimator_
        n_neighbors = best.n_neighbors
        y_pred = best.predict(X_test)
        accuracy  = eval_metrics(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("n_neighbors", n_neighbors)

        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(lr, "model", signature=signature)

        with open("knn_mental_health.pkl", "wb") as file:
            joblib.dump(lr, file)

    dfruns = mlflow.search_runs()
    path2model = dfruns.sort_values("metrics.accuracy", ascending=False).iloc[0]['artifact_uri'].replace("file://","") + '/model' #путь до эксперимента с лучшей моделью
    print(path2model)

