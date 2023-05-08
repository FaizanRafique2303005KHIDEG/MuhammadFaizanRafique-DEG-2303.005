import fire
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



def setup_RandomForest_model(n, d):
    # Here n_estimators parameter specifies the number of decision trees to be created in the forest.
    # max_depth parameter specifies the maximum depth of each tree in the forest.
    RandomForest = RandomForestClassifier(n_estimators=n, max_depth=d)
    return RandomForest


def split_data(df):
    y = df['quality']
    x = df.drop(columns=['quality'])
    
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2)
    
    return X_train, X_test, Y_train, Y_test


def track_with_mlflow(model, X_test, Y_test, mlflow, model_metadata):
    mlflow.log_params(model_metadata)
    mlflow.log_metric("accuracy", model.score(X_test, Y_test))
    mlflow.sklearn.log_model(model, "RandomForest", registered_model_name="sklearn_RandomForest")


def main(file_name: str, max_n: int, max_d: int):
    df = pd.read_csv(file_name,sep=';')

    X_train, X_test, Y_train, Y_test = split_data(df)
    # let's check some other n & d
    n_list = range(1, max_n)
    d_list = range(1, max_d)

    for n in n_list:
        for d in d_list:
            with mlflow.start_run():
                RandomForest_pipe = setup_RandomForest_model(n, d)
                RandomForest_pipe.fit(X_train, Y_train)
                model_metadata = {"n": n, "d": d}
                track_with_mlflow(RandomForest_pipe, X_test, Y_test, mlflow, model_metadata)


if __name__ == "__main__":
    fire.Fire(main)