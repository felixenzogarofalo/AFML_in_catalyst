import pandas as pd
import numpy as np
from ta import *
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# from matplotlib.finance import _candlestick
from matplotlib.dates import date2num
from datetime import datetime

class holder():
    1

# Heiken Ashi Candles
def heikenashi(prices):
    """
    param prices: dataframe de datos OHLC y volumen
    param periods: periodos para los cuales se crearan las velas
    return: velas heiken ashi OHLC
    """
    HAclose = prices[["open", "high", "low", "close"]].sum(axis=1) / 4
    HAopen = HAclose.copy()
    HAopen.iloc[0] = HAclose.iloc[0]

    HAhigh = HAclose.copy()
    HAlow = HAclose.copy()

    for i in range(1, len(prices)):
        HAopen.iloc[i] = (HAopen.iloc[i - 1] + HAclose.iloc[i - 1]) / 2
        HAhigh.iloc[i] = np.array((prices["high"].iloc[i], HAopen.iloc[i], HAclose.iloc[i])).max()
        HAlow.iloc[i] = np.array((prices["low"].iloc[i], HAopen.iloc[i], HAclose.iloc[i])).min()

    df = pd.concat((HAopen, HAhigh, HAlow, HAclose), axis=1)
    df.columns = ["HAopen", "HAhigh", "HAlow", "HAclose"]

    return df


# Quitar tendencia
def detrend(prices, method="difference"):
    """
    :param prices: dataframe de datos OHLC y volumen
    :param method: cadena de texto con el método a utilizar
    :return: pandas DataFrame con datos de cierre sin tendencia
    """
    if method == "difference":
        detrended = pd.DataFrame(data=np.zeros(len(prices)))
        detrended.iloc[0] = prices.close[1] - prices.close[0]
        for i in range(1, len(prices)):
            detrended.iloc[i] = prices.close[i] - prices.close[i-1]
        detrended = detrended.values

    elif method == "linear":
        x = np.arange(0, len(prices))
        y = prices["close"].values

        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y.reshape(-1, 1))

        trend = model.predict(x.reshape(-1, 1))
        trend = trend.reshape((len(prices),))

        detrended = prices.close - trend

    else:
        print("No se ha especificado un método correcto para quitar la tendencia")
        return

    detrended = pd.DataFrame(data=detrended,
                             index=prices.index.tolist(),
                             columns=["Detrended"])

    return detrended

# función de ajuste de la Serie de Expansión de Fourier
def fseries(x, a0, a1, b1, w):
    """

    :param x: valor de tiempo
    :param a0: primer coeficiente de la serie
    :param a1: segundo coeficiente de res = scipy.optimize.curve_fit(fseries, x, y)la serie
    :param b1: tercer coeficiente de la serie
    :param w: frecuencia de la serie
    :return: retorna el valor de la serie de Fourier
    """

    f = a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x)

    return f


# función de ajuste de la Serie de Expansión de Seno
def sseries(x, a0, b1, w):
    """

    :param x: valor de tiempo
    :param a0: primer coeficiente de la serie
    :param b1: tercer coeficiente de la serie
    :param w: frecuencia de la serie
    :return: retorna el valor de la serie de Fourier
    """

    f = a0 + b1 * np.sin(w * x)

    return f

# Función que calcula los coeficientes de la serie de Fourier
def fourier(prices, periods, method="difference"):
    """
    :param prices: OHLC dataframe
    :param periods: lista de periodos para los cuales computar los coeficientes
    :param method: método por el cual quitar la tendencia
    :return: diccionario de coeficientes para los períodos especificados
    """

    results = holder()
    dict = {}

    plot = False

    # Computar los coeficientes

    detrended = detrend(prices, method)

    p0 = None
    for i in range(0, len(periods)):
        coeffs = []

        for j in range(periods[i], len(prices)):
            x = np.arange(0, periods[i])
            y = detrended.iloc[j - periods[i]:j].values.reshape(-1)

            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)

                try:
                    if p0 is None:
                        res, _ = scipy.optimize.curve_fit(fseries, x, y, p0)
                        p0 = res
                    else:
                        res, _ = scipy.optimize.curve_fit(fseries, x, y, p0)
                except (RuntimeError, OptimizeWarning):
                    res = np.empty(4)
                    res[:] = np.NAN

            if plot == True:
                xt = np.linspace(0, periods[i], 100)
                yt = fseries(xt, res[0], res[1], res[2], res[3])

                plt.plot(x, y)
                plt.plot(xt, yt, "r")

                plt.show()

            coeffs = np.append(coeffs, res, axis=0)

        coeffs = np.array(coeffs).reshape((len(coeffs)//(len(res)), len(res)))

        df = pd.DataFrame(coeffs, index=prices.iloc[periods[i]:])

        df.columns = [["a0", "a1", "b1", "w"]]

        df.fillna(method="ffill")

        dict[periods[i]] = df

        results.coeffs = dict

    return results


# Función que calcula los coeficientes de la serie de Sen
def sine(prices, periods, method="difference"):
    """
    :param prices: OHLC dataframe
    :param periods: lista de periodos para los cuales computar los coeficientes
    :param method: método por el cual quitar la tendencia
    :return: diccionario de coeficientes para los períodos especificados
    """

    results = holder()
    dict = {}

    plot = False

    # Computar los coeficientes

    detrended = detrend(prices, method)

    p0 = None
    for i in range(0, len(periods)):
        coeffs = []

        for j in range(periods[i], len(prices)):
            x = np.arange(0, periods[i])
            y = detrended.iloc[j - periods[i]:j].values.reshape(-1)

            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)

                try:
                    res, _ = scipy.optimize.curve_fit(sseries, x, y, method="lm", maxfev=800)
                except (RuntimeError, OptimizeWarning):
                    res = np.empty(3)
                    res[:] = np.NAN
            if plot == True:
                xt = np.linspace(0, periods[i], 100)
                yt = sseries(xt, res[0], res[1], res[2])

                plt.plot(x, y)
                plt.plot(xt, yt, "r")

                plt.show()

            coeffs = np.append(coeffs, res, axis=0)

        coeffs = np.array(coeffs).reshape((len(coeffs)//(len(res)), len(res)))

        df = pd.DataFrame(coeffs, index=prices.iloc[periods[i]:])
        df.columns = [["a0", "b1", "w"]]

        df.fillna(method="ffill")
        dict[periods[i]] = df

    results.coeffs = dict

    return results

def wadl(prices, period):
    """
    Williams Accumulation Distribution Function
    :param prices: dataframe de precios OHLC
    :param period: (list) período para calcular la función
    :return: Lineas de Williams Accumulation Distribution para cada período
    """

    results = holder()
    dict = {}

    WAD = []

    for j in range(period, len(prices)):
        TRH = np.array([prices.high.iloc[j], prices.close.iloc[j-1]]).max()
        TRL = np.array([prices.low.iloc[j], prices.close.iloc[j - 1]]).min()

        if prices.close.iloc[j] > prices.close.iloc[j-1]:
            PM = prices.close.iloc[j] - TRL
        elif prices.close.iloc[j] < prices.close.iloc[j-1]:
            PM = prices.close.iloc[j] - TRH
        elif prices.close.iloc[j] == prices.close.iloc[j-1]:
            PM = 0
        else:
            print(str(prices.close.iloc[j]), str(prices.close.iloc[j-1]))
            print(prices.shape)
            print("Error inesperado")

        AD = PM * prices.volume.iloc[j]

        WAD = np.append(WAD, AD)

    WAD = WAD.cumsum().reshape(-1, 1)

    array = np.empty(shape=(prices.shape[0],))
    array[:] = np.nan
    WADL = pd.DataFrame(data=array,
                        index=prices.index.tolist(),
                        columns=["WAD"])

    WADL.iloc[period:] = WAD
    WADL.fillna(method="bfill")

    return WADL

def create_up_down_dataframe(data,
                             lookback_w=5,
                             lookforward_w=5,
                             up_down_factor=2.0,
                             percent_factor=0.01):
    """
    Crea un DataFrame de pandas que crea un etiqueta cuando el mercado se mueve hacia arriba
    "up_down_factor * percent_factor" en período de "lookfoward_w" mientras que no cae por debajo de "percent_factor"
    en el mismo período
    :param data: DataFrame con los datos
    :param lookback_w: ventana para mirar hacia atrás
    :param lookfoward_w: venta de predicción
    :param up_down_factor: factor de amplificación
    :param percent_factor: porcentaje
    :return: Numpy Array con la característica descrita previamente
    """
    data = data.copy()

    # Crear desplazamientos hacia atrás
    for i in range(lookback_w):
        data["Lookback%s" % str(i+1)] = data.close.shift(i+1)

    # Crear los desplazamientos hacia hacia adelante
    for i in range(lookforward_w):
        data["Lookfoward%s" % str(i+1)] = data.close.shift(-(i+1))

    data.fillna(method="ffill")

    # Ajustar todos estos valores para que sean porcentajes de retorno
    for i in range(lookforward_w):
        data["Lookback%s" % str(i+1)] = data["Lookback%s" % str(i+1)].pct_change() * 100.0
    for i in range(lookforward_w):
        data["Lookfoward%s" % str(i+1)] = data["Lookfoward%s" % str(i+1)].pct_change() * 100.0

    data.fillna(method="ffill")

    # Al utilizar la lógica de esta porción de código se genera un desbalanceo de clases,
    # Que puede verse al plotear la matriz de confución de los resultados.
    """
     up = up_down_factor * percent_factor
    down = percent_factor

    down_cols = [data["Lookfoward%s" % str(i + 1)] > -down
                 for i in range(lookforward_w)]

    up_cols = [data["Lookfoward%s" % str(i + 1)] > up
               for i in range(lookforward_w)]
               
    down_tot = down_cols[0]
    for c in down_cols[1:]:
        down_tot = down_tot & c

    up_tot = up_cols[0]
    for c in up_cols[1:]:
        up_tot = up_tot | c

    data["UpDown"] = down_tot & up_tot
    """

    # En lugar de usar la lógica anterior, se prefiere utilizar como etiqueta la dirección del movimiento
    # de 5 períodos en el futuro.
    n = lookforward_w

    data["Lookfoward" + str(n)].fillna(method="bfill", inplace=True)
    data["Lookfoward" + str(n)].fillna(method="ffill", inplace=True)

    data["UpDown"] = np.sign(data["Lookfoward" + str(n)])
    data["UpDown"] = data["UpDown"].astype(int)
    data["UpDown"].replace(to_replace=0, value=-1, inplace=True)

    features_columns = []
    for i in range(1, lookback_w +1):
        features_columns.append("Lookback" + str(i))

    features = data[features_columns].copy()
    features.fillna(method="bfill", inplace=True)

    return features.values, data["UpDown"].values

def bbands(price, window=None, width=None, numsd=None):
    """
    Devuelve el promedio, banda superior y banda inferior
    :param price: Pandas Dataframe de los precios de cierre
    :param window: número entero que representa la longitud de la ventana
    :param width: número entero representado el ancho fijo de la banda
    :param numsd: número entero representando el número de desviaciones estándar
    :return: DataFrame, Float, Float
    """

    ave = price.rolling(window).mean()
    sd = price.rolling(window).std(ddof=0)
    if width:
        upband = ave * (1+width)
        dnband = ave * (1-width)
        return price, np.round(ave, 8), np.round(upband, 8), np.round(dnband, 8)
    if numsd:
        upband = ave + (sd*numsd)
        dnband = ave - (sd*numsd)
        return price, np.round(ave, 8), np.round(upband, 8), np.round(dnband, 8)

def get_up_cross(df, col):
    """
    Devuelve un DataFrame de pandas con valores correspondientes al cruce de la banda superior
    :param df: DataFrame con los valores de las bandas bollinger
    :param col: columna de precios
    :return: DataFrame
    """
    crit1 = df[col].shift(1) < df.upper.shift(1)
    crit2 = df[col] > df.upper
    return df[col][(crit1) & (crit2)]


def get_down_cross(df, col):
    """
    Devuelve un DataFrame de pandas con valores correspondientes al cruce de la banda inferior
    :param df: DataFrame con los valores de las bandas bollinger
    :param col: columna de precios
    :return: DataFrame
    """
    crit1 = df[col].shift(1) > df.lower.shift(1)
    crit2 = df[col] < df.lower
    return df[col][(crit1) & (crit2)]


def ElliotWaveOsc(df):
    """
    :param df: DataFrame original
    :return: DataFrame con columna de Oscilador Elliot y el Precio base de dicho oscilador
    """
    # Creemos los valores de entrada para el Oscilador de Elliot
    price = (df.high + df.low) / 2
    ma_fast = price.rolling(5).mean().fillna(method="bfill")
    ma_slow = price.rolling(35).mean().fillna(method="bfill")
    elliot_oscillator = ma_fast - ma_slow
    df["OscPrice"] = price
    df["Osc"] = elliot_oscillator

    return df

def ElliotTrendAndWaveCounter(df, window=80, trigger=0.7):
    """
    Esta función calcula la tendencia de la serie temporal, el número de onda
    Elliot en que se encuentra la serie temporal y el valor del indicador de
    compra cuando se cumple cualquiera de las siguientes condiciones:
        1. Se pasa de una onda "negativa", o cero, a una onda 5
        2. Se pasa de una onda 4 a una onda 5
        3. Se pasa de una onda 5 a una onda 3
    :param df: DataFrame con columna de valores de Oscilador Elliot
    :param window: ventana para valor rodante de ElliotOscillator
    :param trigger: porcentaje de retraso para cambio de tendencia
    :return: DataFrame con columnas extra de tendencia, número de onda
                y valor de indicador
    """
    df = ElliotWaveOsc(df)
    df["trend"] = 0
    df["wave"] = 0
    df["ElliotIndicator"] = 0
    trend = 0
    HiOsc = -99999
    HiPrice = -99999
    HiOsc2 = -99999
    HiPrice2 = -99999
    Wave = 0
    WaveCounter = 0
    j = 0  # "j" representará el índice entero del loop
    for i, item in df["trend"].iteritems():
        if df["Osc"].loc[i] == df["Osc"].rolling(window).max().loc[i] and trend == 0:
            trend = 1
            df["trend"].loc[i] = trend
        if df["Osc"].loc[i] == df["Osc"].rolling(window).min().loc[i] and trend == 0:
            trend = -1
            df["trend"].loc[i] = trend
        if df["Osc"].rolling(window).min().loc[i] < 0 and \
                trend == -1 and \
                df["Osc"].loc[i] > 1 * trigger * df["Osc"].rolling(window).min().loc[i]:
            trend = 1
            df["trend"].loc[i] = trend
        if df["Osc"].rolling(window).max().loc[i] > 0 and \
                trend == 1 and \
                df["Osc"].loc[i] < -1 * trigger * df["Osc"].rolling(window).max().loc[i]:
            trend = -1
            df["trend"].loc[i] = trend

        # En este mismo loop etiquetemos el número de onda
        # Para ello debemos garantizar que haya al menos un valor anterior.
        if j > 0:
            # Cuando la tendencia cambie de -1 a 1 etiquetar una onda 3
            # y guardar el oscilador máximo actual y el precio
            if df["trend"].loc[i] == 1 and df["trend"].iloc[j - 1] == -1 and df["Osc"].loc[i] > 0:
                HiOsc = df["Osc"].loc[i]
                HiPrice = df["OscPrice"].loc[i]
                Wave = 3
                df["wave"] = Wave
            # Si la onda 3 y el oscilador hacen un nuevo máximo, guárdarlo.
            if Wave == 3 and HiOsc < df["Osc"].loc[i]:
                HiOsc = df["Osc"].loc[i]
            # Si la onda 3 y el precio hacen un nuevo máximo, guárdarlo.
            if Wave == 3 and HiPrice < df["OscPrice"].loc[i]:
                HiPrice = df["OscPrice"].loc[i]
            # Si nos enconramos en la onda 3 y el oscilador retrocede a cero
            # etiqueta la onda como onda 4
            if Wave == 3 and df["Osc"].loc[i] <= 0 and df["trend"].loc[i] == 1:
                Wave = 4
                df["wave"] = Wave
            # Si nos encontramos en una onda 4 y el oscilador se devuelve por encima
            # cero y hay una ruptura en los precios, entonces etiquetar una onda 5
            # y establecer el segundo conjunto del máximo de oscilador y precio1
            if Wave == 4 and \
                    df["OscPrice"].loc[i] == df["OscPrice"].rolling(5).max().fillna(method="bfill").loc[i] and \
                    df["Osc"].loc[i] >= 0:
                Wave = 5
                df["wave"].loc[i] = Wave
                HiOsc2 = df["Osc"].loc[i]
                HiPrice2 = df["OscPrice"].loc[i]
            if Wave == 5 and HiOsc2 < df["Osc"].loc[i]:
                HiOsc2 = df["Osc"].loc[i]
            # Si el oscilador alcanza un nuevo máximo re-etiquetar como onda 3 y
            # resetear los niveles de la onda 5
            if HiOsc2 > HiOsc and HiPrice2 > HiPrice and Wave == 5 and df["trend"].loc[i] == 1:
                Wave = 3
                df["wave"].loc[i] = Wave
                HiOsc = HiOsc2
                HiPrice = HiPrice2
                HiOsc2 = -99999
                HiPrice2 = -99999
            # Si la tendencia cambia en una onda 5 etiquetar estas como -3 o una
            # onda 3 hacia abajo
            if Wave == 5 and df["trend"].loc[i] == -1 and df["trend"].iloc[j - 1]:
                Wave = -3
                df["wave"].loc[i] = Wave
                HiOsc = -99999
                HiPrice = -99999
                HiOsc2 = -99999
                HiPrice2 = -99999

            # En este mismo loop vamos a definir el valor de indicador también
            if Wave == 3  and df["wave"].iloc[j - 1] <= 0:
                df["ElliotIndicator"].loc[i] = 1
            if Wave == 5 and df["wave"].iloc[j - 1] == 4:
                df["ElliotIndicator"].loc[i] = 1
            if Wave == 3 and df["wave"].iloc[j - 1] == 5:
                df["ElliottIndicator"] = 1
            if df["Osc"].loc[i] < 0:
                df["ElliotIndicator"] = -1
        j += 1

    return df


def stochasticOsillator(df, lookback=14, softened=3, name=""):
    """
    % K = (Cierre Actual - Menor Mínimo) / (Mayor Máximo - Menor Mínimo) * 100
    % D = media móvil simple de "softened" períodos de % K

    Donde:
    Menor Mínimo: es el mínimo de los mínimos en el período de evalución "lookback".
    Mayor Máximo: es el mayor de los máximos en el período de evaluación "lookback".

    :param df: pandas DataFrame con columnas ["open", "high", "low", "close"]
    :param lookback: periodo para evaluar máximos y mínimos de % K
    :param softened: período de la media movil que suaviza a %K
    :return: pandas DataFrame con las columnas de % K y % D
    """
    max = df.high.rolling(window=lookback).max().fillna(method="bfill")
    min = df.low.rolling(window=lookback).min().fillna(method="bfill")

    if name == "":
        kName = "K" + str(lookback) + "_" + str(softened)
        dName = "D" + str(lookback) + "_" + str(softened)
    else:
        kName = name + "K"
        dName = name + "D"

    df[kName] = (df.close - min) / (max - min) * 100
    df[dName] = df[kName].rolling(window=softened).mean().fillna(method="bfill")

    return df


def dobleMomentumStrategy(df, mayor="mayor", minor="minor", threshold=0.2):
    """
    Esta estrategia se basa en crear eventos de trading cuando ocurre una reversión del momentum menor en la zona de
    sobre-compra o sobre-venta, con la condición de que la reversión sea en la misma dirección del momentum mayor.

    :param df: pandas DataFrame conteniendo las columnas de los estocásticos mayor y menor
    :param mayor: nombre del estocástico mayor
    :param minor: nombre del estocástico menor
    :param threshold: umbral que define la zona de sobre compra y sobre venta.
    :return: devuelve un pandas Series binario, positivo cuando se cumplan las condiciones de la estrategia.
    """

    mayorK = df[mayor + "K"]
    mayorD = df[mayor + "D"]
    minorK = df[minor + "K"]
    minorD = df[minor + "D"]

    # Cruce hacia arriba - Condición para entrar en largo
    longCrit1 = minorK.shift(1) < minorD.shift(1)
    longCrit2 = minorK > minorD
    longCrit3 = mayorK > mayorK.shift(1)
    longCrit4 = mayorD > mayorD.shift(1)
    longCrit5 = mayorK < 50
    longCrit6 = minorK < (100 * threshold)
    long = longCrit1 & longCrit2 & longCrit3 & longCrit4 & longCrit5 & longCrit6
    long = long.to_frame()
    long = long.applymap(lambda x: 1 if x else x)
    long = long.applymap(lambda x: 0 if not x else x)

    # Cruce hacia abajo- Condición para entrar en corto
    shortCrit1 = minorK.shift(1) > minorD.shift(1)
    shortCrit2 = minorK < minorD
    shortCrit3 = mayorK < mayorK.shift(1)
    shortCrit4 = mayorD < mayorD.shift(1)
    shortCrit5 = mayorK > 50
    shortCrit6 = minorK > (100 * (1 - threshold))
    short = shortCrit1 & shortCrit2 & shortCrit3 & shortCrit4 & shortCrit5 & shortCrit6
    short = short.to_frame()
    short = short.applymap(lambda x: -1 if x else x)
    short = short.applymap(lambda x: 0 if not x else x)

    strat = np.zeros(shape=df.shape[0])
    strat = pd.DataFrame(data=strat, columns=["strat"])

    strat["long"] = long.values
    strat["short"] = short.values

    strat["strat"] = np.where(strat["long"] == 1, 1, strat["strat"])
    strat["strat"] = np.where(strat["short"] == -1, -1, strat["strat"])

    df["DMStrat"] = strat["strat"].values

    return df
