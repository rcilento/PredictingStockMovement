import pandas as pd
import numpy as np

class StockOperations:

    @staticmethod
    def name_standardizer(df, close_col, high_col, low_col, volume_col, date_col):
        """
        Essa função serve para padronizar os nomes das colunas para a utilização nos passos posteirores

        input:
            dataframe,
            nome da coluna com preços de fechamento,
            nome da coluna com preços mais altos,
            nome da coluna com preços mais baixos,
            nome da coluna com volume de movimentação,
            nome da coluna com data de referencia,

        output:
            dataframe (com nome no padrão definido)
        """
        if close_col != "Close":
            df["Close"] = df[close_col]
            df = df.drop(close_col)
        if high_col != "High":
            df["High"] = df[high_col]
            df = df.drop(high_col)
        if low_col != "Low":
            df["Low"] = df[low_col]
            df = df.drop(low_col)
        if volume_col != "Volume":
            df["Volume"] = df[volume_col]
            df = df.drop(volume_col)

        df["Date"] = pd.to_datetime(df[date_col])
        if date_col != "Date":
            df = df.drop(date_col)
        return df

    @staticmethod
    def drop_extra_cols(df):
        cols_to_drop = [col for col in df.columns if col not in ["Close", "High", "Low", "Volume", "Date"]]
        return df.drop(columns=cols_to_drop, inplace=False)

    @staticmethod
    def drop_raw_cols(df):
        cols_to_drop = ["Close", "High", "Low", "Volume"]
        return df.drop(columns=cols_to_drop, inplace=False)

    @staticmethod
    def na_solver(df, column_name, mode="median"):
        """
        Método de solução de nulos
        :param df: (pandas dataframe) dataframe
        :param column_name: (string) nome da coluna do dataframe o qual a solução será aplicadas
        :param mode: (string) modo da solução ( drop, median, average ou mode)
        :return: retorna dataframe com nulos na coluna selecionada e conforme o modo de solução pedido
        """
        if mode == "drop":
            df = df.drop(df[df[column_name].isnull()].index)
        elif mode == "median":
            df = df.fillna(value={column_name: df[column_name].median()})
        elif mode == "average":
            df = df.fillna(value={column_name: df[column_name].average()})
        elif mode == "mode":
            df = df.fillna(value={column_name: df[column_name].mode()})
        return df

    @staticmethod
    def tech_indicators(df, param_dict):
        initial_cols = list(df.columns)
        macd_parm = param_dict.get("macd", {})
        obv_parm = param_dict.get("obv", {})
        williams_parm = param_dict.get("williams", {})
        proc_parm = param_dict.get("proc", {})
        stoc_parm = param_dict.get("stochastic_oscillator", {})
        sutte_parm = param_dict.get("sutte", {})
        acum_dist_parm = param_dict.get("accumulation_distribution_oscillator", {})
        commodity_parm = param_dict.get("commodity_channel", {})
        momentum_parm = param_dict.get("momentum", {})
        ma_parm = param_dict.get("moving_average", {})
        ema_parm = param_dict.get("exponential_moving_average", {})
        stochastic_d_param = param_dict.get("stochastic_d", {})
        weighted_average_param = param_dict.get("weighted_average", {})

        if macd_parm.get("calculate", False):
            df = StockOperations._generate_macd_col(df, macd_parm.get("compare_macd_with_signal", False))
        if obv_parm.get("calculate", False):
            df = StockOperations._generate_obv_col(df)
        if williams_parm.get("calculate", False):
            df = StockOperations._generate_williams_col(df,  williams_parm.get("periods", []))
        if proc_parm.get("calculate", False):
            df = StockOperations._generate_proc_col(df, proc_parm.get("periods", []))
        if stoc_parm.get("calculate", False):
            df = StockOperations._generate_stochastic_oscillator_col(df, stoc_parm.get("periods", []))
        if sutte_parm.get("calculate", False):
            df = StockOperations._generate_sutte_col(df)
        if acum_dist_parm.get("calculate", False):
            df = StockOperations._generate_accumulation_distribution_oscillator(df, acum_dist_parm.get("periods", []))
        if commodity_parm.get("calculate", False):
            df = StockOperations._generate_commodity_channel_index(df, commodity_parm.get("periods", []))
        if momentum_parm.get("calculate", False):
            df = StockOperations._generate_momentum(df, momentum_parm.get("periods", []))
        if ma_parm.get("calculate", False):
            df = StockOperations._generate_moving_average_comparison(df, ma_parm.get("periods", []))
        if ema_parm.get("calculate", False):
            df = StockOperations._generate_exponential_moving_average_comparison(df, ema_parm.get("periods", []))
        if stochastic_d_param.get("calculate", False):
            df = StockOperations._generate_stochastic_d(df, stochastic_d_param.get("periods", []))
        if weighted_average_param.get("calculate", False):
            df = StockOperations._generate_weighted_moving_average(df, weighted_average_param.get("periods", []))



        return df

    @staticmethod
    def _generate_macd_col(obj, compare_with_signal):
        list_of_needed_means = [9, 12, 26]
        mean_dict = {}
        for mean in list_of_needed_means:
            mean_col = obj.Close.ewm(span=mean, min_periods=mean).mean()
            mean_dict[mean] = mean_col
        obj["macd"] = mean_dict[12] - mean_dict[26]
        obj["macd_signal"] = obj["macd"] / mean_dict[9]
        if compare_with_signal:
            obj["macd_vs_sinal"] = obj["macd_signal"] / obj["macd_signal"]
        return obj

    @staticmethod
    def _generate_obv_col(obj):
        yesterday_close = obj.Close.shift(1)
        obj.loc[obj.Close > yesterday_close, 'aux'] = obj.Volume
        obj.loc[obj.Close <= yesterday_close, 'aux'] = -obj.Volume

        #gains = np.where(obj.Close > yesterday_close, 1, 0)
        #aux = np.where(gains > 0, obj.Volume, -obj.Volume)
        obj["obv"] = pd.Series(obj.aux.cumsum())
        return obj.drop(columns=["aux"])

    @staticmethod
    def _generate_williams_col(obj, periods):

        for period in periods:
            numerador = (
                    obj.High.rolling(window=period, min_periods=period).max()
                    - obj.Close
            )
            denominador = (
                    obj.High.rolling(window=period, min_periods=period).max()
                    - obj.Low.rolling(window=period, min_periods=period).min()
            )
            obj[f"williams_{period}"] = (numerador / denominador) * (-100)

        return obj

    @staticmethod
    def _generate_proc_col(obj, periods):
        for period in periods:
            past_price_col = obj.Close.shift(period)
            obj[f"proc_{period}"] = ((obj.Close - past_price_col) / past_price_col) * 100
        return obj

    @staticmethod
    def _generate_rsi_col(obj, periods):
        for period in periods:
            yesterday_close = obj.Close.shift(1)
            gains = pd.Series(np.where(obj.Close > yesterday_close, obj.Close - yesterday_close, 0))
            losses = pd.Series(np.where(obj.Close < yesterday_close, yesterday_close - obj.Close, 0))
            gain_avg_col = gains.rolling(window=period, min_periods=period).mean()
            loss_avg_col = losses.rolling(window=period, min_periods=period).mean()
            rs = gain_avg_col / loss_avg_col
            rsi = 100 / (1 + rs)
            obj[f"rsi_{period}"] = rsi
        return obj

    @staticmethod
    def _generate_stochastic_oscillator_col(obj, periods):
        for period in periods:
            numerador = (
                    obj.Close
                    - obj.Low.rolling(window=period, min_periods=period).min()
            )

            denominador = (
                    obj.High.rolling(window=period, min_periods=period).max()
                    - obj.Low.rolling(window=period, min_periods=period).min()
            )

            obj[f"stochastic_oscillator_{period}"] = numerador / denominador
        return obj

    @staticmethod
    def _generate_sutte_col(obj):
        past_price_col = obj.Close.shift(1)
        aux = (obj.Close + past_price_col) / 2

        sutteLow = aux + obj.Close - obj.Low
        sutteHigh = aux + obj.High - obj.Close
        sutte = (sutteLow + sutteHigh) / 2
        obj["sutte"] = obj.Close/ sutte
        return obj

    @staticmethod
    def _generate_accumulation_distribution_oscillator(obj, periods):
        for period in periods:
            period_min = obj.Low.rolling(window=period, min_periods=period).min()
            period_max = obj.High.rolling(window=period, min_periods=period).max()
            cmfv = obj.Volume * (((obj.Close - period_min) - (period_max- obj.Close)) / (period_max - period_min))
            obj[f"accumulation_distribution_{period}"] = pd.Series(cmfv.cumsum())
        return obj

    @staticmethod
    def _generate_commodity_channel_index(obj, periods):
        for period in periods:
            typical_price = (
                obj.Low.rolling(window=period, min_periods=period).sum()
                + obj.High.rolling(window=period, min_periods=period).sum()
                + obj.Close.rolling(window=period, min_periods=period).sum()
            )
            ma = typical_price / period
            aux = typical_price - ma
            mean_deviation = aux / period
            obj[f"commodity_channel_{period}"] = aux/(0.015 * mean_deviation)
        return obj

    @staticmethod
    def _generate_momentum(obj, periods):
        for period in periods:
            obj[f"momentum_{period}"] = obj.Close / obj.Close.shift(period)
        return obj

    @staticmethod
    def _generate_weighted_moving_average(obj, periods):
        for period in periods:
            weights = np.array(sorted(range(1, period+1), reverse=True))
            sum_weights = np.sum(weights)
            obj[f'weighted_ma_{period}'] = (
                obj.Close
                .rolling(window=period, center=True, min_periods=period)
                .apply(lambda x: np.sum(weights * x) / sum_weights, raw=False)
             )
        return obj

    @staticmethod
    def _generate_moving_average_comparison(obj, periods):
        for period in periods:
            obj[f"moving_average_comparison_{period}"] = \
                obj.Close / obj.Close.rolling(span=period, min_periods=period).mean()
        return obj

    @staticmethod
    def _generate_exponential_moving_average_comparison(obj, periods):
        for period in periods:
            obj[f"exponential_moving_average_comparison_{period}"] = \
                obj.Close / obj.Close.ewm(span=period, min_periods=period).mean()
        return obj

    @staticmethod
    def _generate_stochastic_d(obj, periods):
        for period in periods:
            k = obj[f"stochastic_oscillator_{period - 1}"]
            for i in range(1, period):
                k += obj[f"stochastic_oscillator_{period - 1}"].shift(i)
            obj[f"stochastic_d_{period}"] = k / period
        return obj

    @staticmethod
    def get_response_variables(df, list_of_days_forward):
        for days_foward in list_of_days_forward:
            # Creating shifted Close price columns in order to compare them later
            close_shifted = (
                df.Close.shift(-days_foward)
                # Setting the shifted columns that could not be generate,
                # because of the df data limitations as -1 for future manipulation
                .fillna(-1)
            )

            # Creating a aux column that will check if the closing price is higher
            # in the shifted column compared with the non-shifted
            raised_aux = np.where(
                close_shifted > df.Close, 1, 0)

            # Creating a response_variable with checking if the stock raised it's value in x days later on
            df[f"response_{days_foward}"] = np.where(
                close_shifted > 0,
                raised_aux.astype(int), np.NaN)

        return df
