import pandas as pd
import HighPerformanceComputing.MultiprocessingAndVectorization as mpv
import DataAnalysis.FinantialDataStructure as fds
import numpy as np


def getDailyVol(close, span0=100):
    """
    Devuelve la volatilidad diaria en una ventana móvil. Es decir, calcula la volatilidad del valor de cierre i-ésimo
    con respecto al valor de cierre correspondiente al día anterior del valor actual.
    :param close: pandas DataFrame con valores de cierre
    :param span0: especifica el decaimiento de alfa de la ventana móvil ponderada exponencialmente.
                    alfa = 2 / (span + 1), para span >= 1
    :return: pandas Series
    """
    # Volatilidad diaria, re-indexada a cierre
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0-1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # Retornos diarios
    df0 = df0.ewm(span=span0).std()  # EWM: Exponentially Weighted Windows
    df0 = df0.fillna(method="bfill")
    return df0


def getVol(close, span0=100, delta_time = pd.Timedelta(days=1)):
    """
    Devuelve la volatilidad según el rango de tiempo especificado en una ventana móvil. Es decir, calcula la
    volatilidad del valor de cierre i-ésimo con respecto al valor de cierre correspondiente al valor anterior
    del valor actual.
    :param close: pandas DataFrame con valores de cierre
    :param span0: especifica el decaimiento de alfa de la ventana móvil ponderada exponencialmente.
                    alfa = 2 / (span + 1), para span >= 1
    :return: pandas Series
    """
    # Volatilidad diaria, re-indexada a cierre
    df0 = close.index.searchsorted(close.index - delta_time)
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0-1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # Retornos diarios
    df0 = df0.ewm(span=span0).std()  # EWM: Exponentially Weighted Windows
    df0 = df0.fillna(method="bfill")
    return df0


def applyTpS1onT1(close, events, TpSl, molecule):
    """
    Implementa el método de la triple barrera para etiquetar los datos.
    :param close: Series pandas de precios
    :param events: Un DataFrame de pandas con las siguientes columnas:
                    t1: El timestamp de la barrera vertical. Cuando el valor es np.nan, no habrá barrera vertical
                    trgt: el grosor unitario de las barreras horizontales ("target")
    :param TpSl: Una lista de dos valores decimales no negativos:
                    TpSl[0]: factor que multiplica "trgt" para configurar el ancho de la barrera superior.
                            Si es 0 no habrá barrera superior
                    TpSl[1]: factor que multiplica "trgt" para configurar el ancho de la barrera inferior.
                            Si es 0 no habrá barrera inferior
    :param molecule: Una lista con el sub-conjunto de índices de evento que serán procesados a la vez.
    :return: DataFrame de pandas que contiene los timestamps en los cuales cada barrera es tocada.
    """
    # Aplicar Stop Loss / Take Profit, si toman lugar antes de t1 (que es el fin del evento)
    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)
    if TpSl[0] > 0:
        tp = TpSl[0] * events_["trgt"]  # tp: Take Profit o Toma de Ganancias
    else:
        tp = pd.Series(index=events.index)  # NaNs
    if TpSl[1] > 0:
        sl = - TpSl[1] * events_["trgt"]  # sl: Stop Loss o Detención de Pérdidas
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events_["t1"].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1]  # Camino de precios
        df0 = (df0 / close[loc] - 1) * events_.at[loc, "side"]  # Camino de retornos
        out.loc[loc, "sl"] = df0[df0 < sl.loc[loc]].index.min()  # Toca primero el Stop Loss
        out.loc[loc, "tp"] = df0[df0 > tp.loc[loc]].index.min()  # Toca primero el Take Profit    print(out)
    return out


def getEvents(close, tEvents, TpSl, trgt, minRet, numThreads, t1=False, side=None):
    """
    Ubica el timestamp del primer toque de la barrera
    :param close: Serie de precios pandas
    :param tEvents: Pandas time-index conteniendo los timestamps que serán introducidos cada barrera triple.
                    Estos son los timestamps seleccionados por FinantialDataStructure.getTEvents()
    :param TpSl: lista de decimales no-negativo que configuran el ancho de las dos barreras
    :param trgt: Una Series de pandas de objetivos ("targets"), expresados en términos de retornos absolutos.
    :param minRet: El retorno mínimo objetivo requerido para correr la búsqueda de la barrera triple.
    :param numThreads: Número de hilos utilizados por la función actualmente.
    :param t1: Una serie de pandas con los timestamps de las barreras verticales.
    :param side: Pandas DataFrame con el lado de la apuesta.
    :return:
    """
    # 1) Obtener el objetivo
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]  # minRet
    trgt.columns = ["trgt"]

    # 2) Obtener t1 (máximo tiempo para el cruce de barras)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    # 3) Desde el objeto eventos aplicar Stop Loss en t1
    if side is None:
        side_ = pd.Series(1., index=trgt.index)
        TpSl_ = [TpSl[0], TpSl[0]]
    else:
        side_ = side.loc[trgt.index]
        TpSl_ = TpSl[:2]

    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1)

    # df0 = applyTpS1onT1(close=close, events=events, TpSl=TpSl_, molecule=tEvents)

    df0 = mpv.mpPandasObj(func=applyTpS1onT1,
                          pdObj=("molecule", events.index),
                          numThreads=numThreads,
                          close=close,
                          events=events,
                          TpSl=TpSl_)

    # Este el código original, pero no elimina las filas con NaNs en "ret"
    # Este sería el caso cuando el retorno obtenido está por debajo del retorno mínimo especificado en los
    # parámetros de la función
    events["t1"] = df0.dropna(how="all").min(axis=1)  # pd.min ignorando NaN

    # Se elimina las filas con NaN debido a que no superan el umbral de retorno mínimo.
    events = events.dropna(how="any")

    if side is None:
        events = events.drop("side", axis=1)

    events_ = events.copy()

    return events_


def getT1(close, tEvents, numDays):
    """
    Ubica los índices donde es tocado la barrera vertical
    :param close: DataFrame con los precios de cierre
    :param tEvents: DataFrame con los Datetime de cada evento
    :param numDays: Número de días que definen la barrera vertical
    :return: DataFrame con Datetime donde se toca la barrera vertical
    """
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]])  # NaNs al final
    return t1


def getBins(events, close, vb_offset=None):
    """
    Crea las etiquetas
    :param events: DataFrame de eventos
    :param close: DataFrame con precios de cierre
    :param vb_offset: Longitud de barrera vertical usado para crear los valores de t1
    :return:
    """
    # 1) Precios alineados con eventos
    events_ = events.dropna(subset=["t1"])
    px = events_.index.union(events_["t1"].values).drop_duplicates()
    px = close.reindex(px, method="bfill")
    # 2) Crear el objeto de salida
    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].values].values / px.loc[events_.index] - 1
    if "side" in events_:
        out["ret"] *= events_["side"]  # Meta-etiquetado
    out["bin"] = np.sign(out["ret"])  # Dirección
    # Descomentar estas líneas para agregar una etiqueta "0" cuando no haya retorno positivo
    # if "side" in events_:
    #     out.loc[out["ret"] <= 0, "bin"] = 0  # Meta-etiquetado

    return out


def dropLabels(bin, minPtc=0.05):
    """
    Quita las etiquetas que aparezcan menos del porcentaje requerido
    :param bin: DataFrame de etiquetas procesadas
    :param minPtc: Porcentaje mínimo de etiqueta
    :return: DataFrame con etiquetas editadas
    """
    # Aplica pesos, elimina etiquetas sin muestras suficientes
    while True:
        df0 = bin["bin"].value_counts(normalize=True)
        if df0.min() > minPtc or df0.shape[0] < 3:
            break
        print("Dropped Label", df0.argmin(), df0.min())
        bin = bin[bin["bin"] != df0.argmin()]
    return bin


def addVerticalBarrier(tEvents, close, delta_time):
    t1 = close.index.searchsorted(tEvents + delta_time)
    t1 = t1[t1 < close.shape[0]]
    t1 = (pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]))
    return t1
