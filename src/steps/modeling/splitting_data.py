from src.utils import save_info
import pandas as pd


def splitting_data(master: pd.DataFrame):
    response_cols = [col for col in master.columns if "response" in col]
    features_cols = [col for col in master.columns if col not in response_cols]

    test_data = master.tail(int(len(master.index) * 0.2))
    train_data = master.drop(test_data.index)

    dfs = {
        "X_train": train_data[features_cols],
        "X_test": test_data[features_cols],
        "y_train_df": train_data[response_cols],
        "y_test_df": test_data[response_cols]
    }
    save_info(f"Train: {len(train_data)} linhas - periodo -> {min(dfs['X_train'].index)} - {max(dfs['X_train'].index)}",
              "text", "general_info")
    save_info(f"Test: {len(test_data)} linhas - periodo ->  {min(dfs['X_test'].index)} - {max(dfs['X_test'].index)}",
              "text", "general_info")

    for df_name, df in dfs.items():
        save_info(df, "dataframe", df_name)
        save_info(df.describe(), "dataframe", df_name + "_describe")

    return dfs["X_train"], dfs["y_train_df"], dfs["X_test"], dfs["y_test_df"]
