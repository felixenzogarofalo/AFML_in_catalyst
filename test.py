import matplotlib.pyplot as plt
from catalyst import run_algorithm
from catalyst.api import symbols
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from features import create_up_down_dataframe as updown
from features import ElliotTrendAndWaveCounter as elliotIndicator
import DataAnalysis.FinantialDataStructure as fds
import pandas as pd
from logbook import Logger
from sklearn.externals import joblib
from sklearn import preprocessing
import pickle

NAMESPACE = 'Deep Deterministic Policy Gradient'
log = Logger(NAMESPACE)


def initialize(context):
    # Lista de activos
    context.assets = ["btc_usdt"]
    context.symbols = symbols("btc_usdt")

    # Definir carácteristicas
    context.row_features = ["open", "high", "low", "close", "volume"]
    context.features = ["open", "high", "low", "close", "volume"]

    context.render = True

    context.bar_period = 30
    context.lookback_w = 30
    context.lookforward_w = 5
    context.vol_freq = 10


def handle_data(context, data):
    if context.render:
        h1 = data.history(context.symbols,
                          context.row_features,
                          bar_count=10000,
                          frequency="30T",
                          )

        h1 = h1.swapaxes(2, 0)

        # Por ahora se está trabajando con un solo activo
        h1 = h1.iloc[0]
        # vbs = fds.VolumeBarSeries(df=h1, vol_frequency=context.vol_freq).process_data()

        df = elliotIndicator(h1, window=80, trigger=0.7)

        df_down = h1.copy()
        df_down[["open", "high", "low", "close"]] = df_down[["open", "high", "low", "close"]] * (-1)
        df_down = elliotIndicator(df_down)

        df["ElliotIndicator"].plot(legend=True)
        df["wave"].plot(legend=True)
        close = df["close"] / df["close"].iloc[0]
        close.plot(legend=True)
        plt.show()

        df_down["ElliotIndicator"].plot(legend=True)
        df_down["wave"].plot(legend=True)
        down_close = df_down["close"] / df_down["close"].iloc[0]
        down_close.plot(legend=True)
        plt.show()

        context.render = False


def analyze(context, perf):
    pass


def get_data(context, data_, window):
    # Crear ventana de datos.
    h1 = data_.history(context.symbols,
                       context.row_features,
                       bar_count=window,
                       frequency=str(context.bar_period) + "T",
                       )

    h1 = h1.swapaxes(2, 0)

    # Por ahora se está trabajando con un solo activo
    data = h1.iloc[0]

    features, label = updown(data=data,
                             lookback_w=context.lookback_w,
                             lookforward_w=context.lookforward_w,
                             up_down_factor=2.0,
                             percent_factor=0.01)

    return features, label


if __name__ == '__main__':
    run_algorithm(
        capital_base=1000,
        data_frequency='minute',
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        exchange_name='poloniex',
        algo_namespace=NAMESPACE,
        quote_currency='usd',
        start=pd.to_datetime('2018-1-1', utc=True),
        end=pd.to_datetime('2018-1-1', utc=True),
    )
