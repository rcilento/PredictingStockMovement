from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def modeling(X_train, y_train_df, X_test, y_test_df):

    results_dict = {
        "SVM": {
            "model": Pipeline([('scaler', StandardScaler()), ('model', LinearSVC(random_state=0, max_iter=1000))]),
            "F1": [],
            "ACCURACY": [],

            "PRECISION": [],
            "RECALL": []
        },
        "RANDOM FOREST": {
            "model": Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier(n_estimators=1001, random_state=0, min_samples_leaf=300))]),
            "AUC": [],
            "F1": [],
            "ACCURACY": [],
            "PRECISION": [],
            "RECALL": []
        }
    }

    response_cols = list(y_test_df.columns)
    for model_name in results_dict.keys():

        model = results_dict[model_name]["model"]
        for response_period in response_cols:

            y_train = y_train_df[response_period]
            y_test = y_test_df[response_period]

            print("Response:", response_period)

            model_fit = model.fit(X_train, y_train)

            train_preds = model_fit.predict(X_train)
            test_preds = model_fit.predict(X_test)
            if model_name == "RANDOM FOREST":
                test_probs = model_fit.predict_proba(X_test)[:, 1]
                train_probs = model_fit.predict_proba(X_train)[:, 1]

            tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
            if "AUC" in list(results_dict[model_name].keys()):
                results_dict[model_name]["AUC"].append(roc_auc_score(y_test, test_probs))
            results_dict[model_name]["F1"].append(f1_score(y_test, test_preds))
            results_dict[model_name]["ACCURACY"].append(accuracy_score(y_test, test_preds))
            results_dict[model_name]["RECALL"].append(tp / (tp + fp))
            results_dict[model_name]["PRECISION"].append(tp / (tp + fn))
    return results_dict
