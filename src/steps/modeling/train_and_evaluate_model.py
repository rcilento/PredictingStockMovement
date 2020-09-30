from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
import pandas as pd
from src.utils import save_info

def train_and_evaluate_model(X_train, y_train_df, X_test, y_test_df, predictor, calculate_feature_importance=False,
                             calculate_coeffs=False):
    results_dict = {
            "TRAIN_F1": [],
            "TEST_F1": [],
            "TRAIN_ACCURACY": [],
            "TEST_ACCURACY": [],
            "TRAIN_PRECISION": [],
            "TEST_PRECISION": [],
            "TRAIN_RECALL": [],
            "TEST_RECALL": [],
            "TRAIN_HOLDOUT_F1": [],
            "TRAIN_HOLDOUT_ACCURACY": [],
            "TRAIN_HOLDOUT_RECALL": [],
            "TRAIN_HOLDOUT_PRECISION": []
        }
    if calculate_feature_importance:
        results_dict["feature_importance"] = []
    if calculate_coeffs:
        results_dict["coeffs"] = []

    response_cols = list(y_test_df.columns)
    model = Pipeline([('scaler', MinMaxScaler()), ('model', predictor)])

    for response_period in response_cols:

        y_train = y_train_df[response_period]
        y_test = y_test_df[response_period]

        print("Response:", response_period)

        model_fit = model.fit(X_train, y_train)

        holdout_scores = cross_validate(model, X_train, y_train, scoring=["accuracy", "f1", "precision", "recall"],
                                        cv=5, n_jobs=-1)

        if calculate_feature_importance:
            feat_imp = pd.Series(data=list(model_fit["model"].feature_importances_), index=list(X_train.columns),
                                 name=response_period)
            results_dict["feature_importance"].append(feat_imp)

        if calculate_coeffs:
            if calculate_feature_importance:
                best_predictors = list(feat_imp.sort_values(ascending=False).index)[0:9]
                aux_model = Pipeline([('scaler', MinMaxScaler()), ('model', LogisticRegression())])
                aux_model_fit = aux_model.fit(X_train[best_predictors], y_train)
            else:
                aux_model_fit = model_fit
                best_predictors = X_train.columns
            coefs = pd.Series(data=list(aux_model_fit["model"].coef_[0]), index=best_predictors, name=response_period)
            results_dict["coeffs"].append(coefs)


        def get_mean(lista):
            return sum(lista)/len(lista)

        results_dict["TRAIN_HOLDOUT_F1"].append(get_mean(holdout_scores["test_f1"]))
        results_dict["TRAIN_HOLDOUT_ACCURACY"].append(get_mean(holdout_scores["test_accuracy"]))
        results_dict["TRAIN_HOLDOUT_RECALL"].append(get_mean(holdout_scores["test_recall"]))
        results_dict["TRAIN_HOLDOUT_PRECISION"].append(get_mean(holdout_scores["test_precision"]))

        train_pred = model_fit.predict(X_train)
        tn, fp, fn, tp = confusion_matrix(y_train, train_pred).ravel()
        results_dict["TRAIN_F1"].append(f1_score(y_train, train_pred))
        results_dict["TRAIN_ACCURACY"].append(accuracy_score(y_train, train_pred))
        results_dict["TRAIN_RECALL"].append(recall_score(y_train, train_pred))
        results_dict["TRAIN_PRECISION"].append(precision_score(y_train, train_pred))

        test_pred = model_fit.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
        results_dict["TEST_F1"].append(f1_score(y_test, test_pred))
        results_dict["TEST_ACCURACY"].append(accuracy_score(y_test, test_pred))
        results_dict["TEST_RECALL"].append(recall_score(y_test, test_pred))
        results_dict["TEST_PRECISION"].append(precision_score(y_test, test_pred))

    return results_dict
