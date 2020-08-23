param_dict = {
    "macd": {
        "calculate": True,
        "compare_macd_with_signal": True
    },
    "obv": {
        "calculate": True
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
        "calculate": False,
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

file_path = "data/BVSP.csv"

response_period_experiment_list = list(range(1, 101))

filter_date_greater_than = "1998-01-01"
test_period = "2016-01-01"

pearson_r_threshold = 0.8

experiment_name = "02"
