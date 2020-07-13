import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression


class TimeSeries():
    def __init__(self):
        self.time_series_df = pd.read_csv("../../data/passagierzahlen.csv", skiprows=1, parse_dates=["Monat/Jahr"],
                                          index_col="Monat/Jahr",
                                          dayfirst=True)
        # Loading data
        self.time_series_df["Passagierzahlen"] = self.time_series_df["Passagierzahlen"].map(
            lambda x: x.replace(".", "")).astype(np.int64)

        # Time series analysis
        s_decomp = seasonal_decompose(self.time_series_df['Passagierzahlen'], model='multiplicative',
                                      extrapolate_trend='freq')
        self.s_decomp_nc = seasonal_decompose(self.time_series_df['Passagierzahlen'][:-2], model='multiplicative',
                                              extrapolate_trend='freq')

        # Regression on trend: With Corona data
        x = np.linspace(0, s_decomp.trend.values.shape[0], s_decomp.trend.values.shape[0]).reshape((-1, 1))
        y = np.reshape(s_decomp.trend.values, (-1, 1))
        reg_trend = LinearRegression().fit(x, y)

        # Regression on trend: Without Corona data
        x_nc = np.linspace(0, self.s_decomp_nc.trend.values.shape[0], self.s_decomp_nc.trend.values.shape[0]).reshape(
            (-1, 1))
        y_nc = np.reshape(self.s_decomp_nc.trend.values, (-1, 1))
        reg_trend_nc = LinearRegression().fit(x_nc, y_nc)

    def get_residuums_dates(self, *, spread=None):
        values = self.get_residuums(spread=spread)
        dates = self.time_series_df.index
        dates = [d.strftime('%d.%m.%Y') for d in dates]
        return list(zip(values, dates))

    def get_residuums(self, *, spread=None):
        values = self.s_decomp_nc.resid.values

        if spread is None:
            return values
        else:
            return list(self._make_binary(values, 1 + spread, 1 - spread))

    def _normalize(self, values):
        values = values - 1
        maximum = np.max(values)
        minimum = np.min(values)
        diviser = maximum - minimum
        values = (values - minimum) * 2
        values = values / diviser
        values = -1 + values
        return values

    def _make_binary(self, values, upper_bound, lower_bound):
        for v in values:
            if v >= upper_bound:
                yield 1
            elif v <= lower_bound:
                yield -1
            else:
                yield 0

    def plot_results(self, list):
        # Plotting
        # ts.plot()
        # plt.figure()
        # s_decomp.seasonal.plot(title="Seasonal")
        # plt.figure(figsize=(19, 4))
        plt.plot(list, color='blue')
        plt.figure()
        # plt.title(f'Spread: {spread}')
        # plt.plot(s_decomp.trend.index, s_decomp.trend.values)
        # plt.plot(s_decomp.trend.index, reg_trend.predict(x), label="Corona included")
        # plt.plot(s_decomp_nc.trend.index, reg_trend_nc.predict(x_nc), label="Corona excluded")
        # plt.title("Linear Regression on Trend")
        # plt.legend()
        plt.show()
