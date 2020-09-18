from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from src.utils import save_info

def train_and_evaluate_model(X_train, y_train_df, X_test, y_test_df, predictor, calculate_feature_importance=False):
    results_dict = {
            "TRAIN_F1": [],
            "TEST_F1": [],
            "TRAIN_ACCURACY": [],
            "TEST_ACCURACY": [],
            "TRAIN_PRECISION": [],
            "TEST_PRECISION": [],
            "TRAIN_RECALL": [],
            "TEST_RECALL": [],
        }
    if calculate_feature_importance:
        results_dict["feature_importance"] = []

    response_cols = list(y_test_df.columns)
    model = Pipeline([('scaler', StandardScaler()), ('model', predictor)])

    for response_period in response_cols:

        y_train = y_train_df[response_period]
        y_test = y_test_df[response_period]

        print("Response:", response_period)

        model_fit = model.fit(X_train, y_train)

        holdout_scores = cross_validate(model, X_train, y_train, scoring=["accuracy", "f1", "precision", "recall"], cv=5)

        if calculate_feature_importance:
            results_dict["feature_importance"].append(dict(zip(X_train.columns, model_fit["model"].feature_importances_)))

        results_dict["TRAIN_F1"].append(sorted(holdout_scores["test_f1"])[2])
        results_dict["TRAIN_ACCURACY"].append(sorted(holdout_scores["test_accuracy"])[2])
        results_dict["TRAIN_RECALL"].append(sorted(holdout_scores["test_recall"])[2])
        results_dict["TRAIN_PRECISION"].append(sorted(holdout_scores["test_precision"])[2])

        test_pred =  model_fit.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
        results_dict["TEST_F1"].append(f1_score(y_test, test_pred))
        results_dict["TEST_ACCURACY"].append(accuracy_score(y_test, test_pred))
        results_dict["TEST_RECALL"].append(tp / (tp + fp))
        results_dict["TEST_PRECISION"].append(tp / (tp + fn))

    return results_dict
