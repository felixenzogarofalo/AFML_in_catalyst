import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def getWeights(d, size):
    # Umbral > 0 elimina pesos insignificantes
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def plotWeights(dRange, nPlots, size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = getWeights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
        w = w.join(w_, how="outer")
    ax = w.plot()
    ax.legend(loc="upper left")
    plt.show()
    return


def fracDiff(series, d ,thres=.01):
    """
    Incrementando el ancho de la ventana, con tratamiento de NaNs
    Nota 1: para thres = 1, no se pasa por alto nada
    Nota 2: d puede ser cualquier fracional positivo, no necesariamente entre [0, 1].
    :param series:
    :param d:
    :param thres:
    :return:
    """
    # 1) Computar los pesos para la serie más larga
    w = getWeights(d, series.shape[0])
    # 2) Determinar los cálculos iniciales que se pasarán por alto basado en el umbral de
    # pérdida de peso
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    # 3) Aplicar pesos a los valores
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method="ffill").dropna(), pd.Series()
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc, name]): continue # Excluir NaNs
            df_[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def fracDiff_FFD(series, d, thres=1e-5):
    """
    Ancho de ventana constante (nueva solución)
    Nota 1: thres determina el peso de recorte para la ventana
    Nota 2: d puede ser cualquier fracional positivo, no necesariamente entre [0, 1].
    :param series:
    :param d:
    :param thres:
    :return:
    """
    # 1) Computar los pesos para la serie más larga
    w = getWeights(d, thres)
    width = len(w) - 1
    # 2) Aplicar pesos a valores
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method="ffil").dropna(), pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]): continue # Excluir NaNs
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


# Hallar el mínimo "d" que pasa la prueba ADF
def plotMinFFD():
    path = "./"
    instName = "ES1_Index_Method12"
    out = pd.DataFrame(columns=["adfStat",
                                "pVal",
                                "lags",
                                "nObs",
                                "95% conf",
                                "corr"])
    df0 = pd.read_csv(path + instName + ".csv", index_col=0, parse_date=True)
    for d in np.linspace(0,1,11):
        df1 = np.log(df0[["Close"]]).resample("1D").last()  # Llevar a observaciones diarias
        df2 = fracDiff_FFD(df1, d, thres=0.01)
        corr = np.corrcoef(df1.loc[df2.index, "Close"], df2["Close"])[0, 1]
        df2 = adfuller(df2["Close"], maxlag=1, regression="c", autolag=None)
        out.loc[d] = list(df2[:4]) + [df2[4]["5%"]] + [corr]
    out.to_csv(path + instName + "_testMinFFD.csv")
    out[["adfStat", "corr"]].plot(secondary_y="adfStat")
    plt.axhline(out["95% conf"].mean(), linewidth=1, color="r", linestyle="dotted")
    plt.savefig(path + instName + "_testMinFFD.png")
    return


if __name__ == "__main__":
    plotMinFFD()




























