def matrix_correlation_filter(X, y=None, method="pearson", min_period=1, threshold=0.8):
    import pandas as pd

    if isinstance(y, pd.Series):
        df = pd.concat([X, y], axis="columns")
        response_var = y.name
        corr_df = df.corr(method, min_period)
        temp = corr_df.sort_values(response_var)
        cols_to_analyze = [col for col in temp[response_var].index]
    else:
        df = X
        # a lista de colunas entra invertida para que possa priorizar manter as variaveis que vem primeiro no dataframe
        cols_to_analyze = [col for col in df.columns][::-1]
        corr_df = df.corr(method, min_period)

    excluded_cols = []
    vars_to_be_analyzed = len(cols_to_analyze)
    for col in cols_to_analyze:
        if col not in corr_df.columns:
            # colunas excluidas por não terem min_period suficiente
            vars_to_be_analyzed -= 1
            excluded_cols += [col]
            continue

        temp = corr_df.drop(col)
        temp = temp[temp[col].abs() > threshold]
        if len(temp):
            corr_df = corr_df.drop(col, axis=0)
            excluded_cols += [col]

        vars_to_be_analyzed -= 1

    return X.drop(excluded_cols, axis=1)
