from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def modeling(X_train, y_train_df, X_test, y_test_df):

    results_dict = {
        "SVM": {
            "model": Pipeline([('scaler', StandardScaler()), ('model', LinearSVC(random_state=0, max_iter=1000))]),
            "param_grid": {"model__C": [1, 2, 5, 7, 10, 20]},
            "best_model": [],
            "TRAIN_F1": [],
            "TEST_F1": [],
            "TRAIN_ACCURACY": [],
            "TEST_ACCURACY": [],
            "TRAIN_PRECISION": [],
            "TEST_PRECISION": [],
            "TRAIN_RECALL": [],
            "TEST_RECALL": [],
        },
        "RANDOM FOREST": {
            "model": Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier(n_estimators=101, random_state=0))]),
            "param_grid": {"model__min_samples_leaf": [50, 100, 150, 200, 300]},
            "best_model": [],
            "TRAIN_F1": [],
            "TEST_F1": [],
            "TRAIN_ACCURACY": [],
            "TEST_ACCURACY": [],
            "TRAIN_PRECISION": [],
            "TEST_PRECISION": [],
            "TRAIN_RECALL": [],
            "TEST_RECALL": [],
            "feature_importance": [],
        }
    }

    response_cols = list(y_test_df.columns)
    for model_name in results_dict.keys():

        model = results_dict[model_name]["model"]
        for response_period in response_cols:

            y_train = y_train_df[response_period]
            y_test = y_test_df[response_period]

            print("Response:", response_period)

            trained_grid = GridSearchCV(model, results_dict[model_name]["param_grid"], cv=5).fit(X_train, y_train)

            model_fit = trained_grid.best_estimator_

            results_dict[model_name]["best_model"].append(trained_grid.best_params_)

            if model_name == "RANDOM FOREST":
                results_dict["RANDOM FOREST"]["feature_importance"].append(dict(zip(X_train.columns, model_fit["model"].feature_importances_)))

            preds = {
                "TRAIN": model_fit.predict(X_train),
                "TEST": model_fit.predict(X_test),
            }
            for public, pred in preds.items():
                real = y_train if public == "TRAIN" else y_test
                tn, fp, fn, tp = confusion_matrix(real, pred).ravel()
                results_dict[model_name][f"{public}_F1"].append(f1_score(real, pred))
                results_dict[model_name][f"{public}_ACCURACY"].append(accuracy_score(real, pred))
                results_dict[model_name][f"{public}_RECALL"].append(tp / (tp + fp))
                results_dict[model_name][f"{public}_PRECISION"].append(tp / (tp + fn))
    return results_dict
