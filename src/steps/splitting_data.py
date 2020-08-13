def splitting_data(stock, test_period, features_cols, response_cols, experiment_config):
    train_data = (
        stock
            .loc[stock['Date'] < test_period]
            .drop(columns="Date", inplace=False)
            .astype("float64")
    )

    test_data = (
        stock
            .loc[stock['Date'] >= test_period]
            .drop(columns="Date", inplace=False)
            .astype("float64")
    )
    X_train = train_data[features_cols]
    y_train_df = train_data[response_cols]
    X_test = test_data[features_cols]
    y_test_df = test_data[response_cols]

    experiment_config["test_period"] = test_period
    experiment_config["train_size"] = len(X_train)
    experiment_config["test_size"] = len(X_test)
    experiment_config["describe_X_train"] = X_train.describe()
    experiment_config["describe_X_test"] = X_test.describe()
    experiment_config["describe_y_train"] = y_train_df.describe()
    experiment_config["describe_y_test"] = y_test_df.describe()

    return X_train, y_train_df, X_test, y_test_df, experiment_config
