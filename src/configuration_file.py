from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier

PARAM_DICT = {
    "macd": {
        "calculate": True,
        "compare_macd_with_signal": True
    },
    "obv": {
        "calculate": False
    },
    "williams": {
        "calculate": True,
        "periods": [2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    },
    "proc": {
        "calculate": True,
        "periods": [2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    },
    "stochastic_oscillator": {
        "calculate": True,
        "periods": [2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    },
    "sutte": {
        "calculate": True,
    },
    "accumulation_distribution_oscillator": {
        "calculate": True,
        "periods": [2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    },
    "commodity_channel": {
        "calculate": True,
        "periods": [2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    },
    "momentum": {
        "calculate": True,
        "periods": [2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    },
    "stochastic_d": {
        "calculate": True,
        "periods": [3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    },

    "weighted_average": {
        "calculate": True,
        "periods": [3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    },
}

PREDICTORS = {
    "SVM C=0.2": LinearSVC(random_state=0, max_iter=100000, C=0.2),
    "SVM C=0.4": LinearSVC(random_state=0, max_iter=100000, C=0.4),
    "SVM C=0.5": LinearSVC(random_state=0, max_iter=100000, C=0.5),
    "SVM C=0.6": LinearSVC(random_state=0, max_iter=100000, C=0.6),
    "SVM C=0.8": LinearSVC(random_state=0, max_iter=100000, C=0.8),
    "SVM C=1": LinearSVC(random_state=0, max_iter=100000, C=1),
    "SVM C=2": LinearSVC(random_state=0, max_iter=100000, C=2),
    "SVM C=4": LinearSVC(random_state=0, max_iter=100000, C=4),
    "SVM C=6": LinearSVC(random_state=0, max_iter=100000, C=6),
    "RANDOM FOREST min_samples_leaf=0.08": RandomForestClassifier(n_estimators=501, random_state=42,
                                                                  min_samples_leaf=0.08, n_jobs=-1),
    "RANDOM FOREST min_samples_leaf=0.10": RandomForestClassifier(n_estimators=501, random_state=42,
                                                                  min_samples_leaf=0.10, n_jobs=-1),
    "RANDOM FOREST min_samples_leaf=0.12": RandomForestClassifier(n_estimators=501, random_state=42,
                                                                  min_samples_leaf=0.12, n_jobs=-1),
    "RANDOM FOREST min_samples_leaf=0.14": RandomForestClassifier(n_estimators=501, random_state=42,
                                                                  min_samples_leaf=0.14, n_jobs=-1),
    "RANDOM FOREST min_samples_leaf=0.16": RandomForestClassifier(n_estimators=501, random_state=42,
                                                                  min_samples_leaf=0.16, n_jobs=-1),
    "RANDOM FOREST min_samples_leaf=0.18": RandomForestClassifier(n_estimators=501, random_state=42,
                                                                  min_samples_leaf=0.18, n_jobs=-1),
    "RANDOM FOREST min_samples_leaf=0.20": RandomForestClassifier(n_estimators=501, random_state=42,
                                                                  min_samples_leaf=0.2, n_jobs=-1),
    "RANDOM FOREST min_samples_leaf=0.22": RandomForestClassifier(n_estimators=501, random_state=42,
                                                                  min_samples_leaf=0.22, n_jobs=-1),
    "Dummy Classifier": DummyClassifier(random_state=42, strategy="stratified")
}

FILE_PATH = "data/BVSP.csv"

RESPONSE_PERIOD_EXPERIMENT_LIST = list(range(1, 101))

ONLY_DAYS_WITH_VOLUME = True

FILTER_DATE_GREATER_THAN = "2006-08-01"

PEARSON_R_THRESHOLD = 2

EXPERIMENT_NAME = "17"
