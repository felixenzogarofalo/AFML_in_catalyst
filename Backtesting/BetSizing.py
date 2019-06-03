from scipy.stats import norm
import pandas as pd
from SupervisedLearningIntraday.HighPerformanceComputing.MultiprocessingAndVectorization import mpPandasObj


def getSignal(events, stepSize, prob, pred, numClasses, numThreads, **kargs):
    """
    :param events: lista de eventos indexados con Datetime
    :param stepSize: paso de discretización de la señal
    :param prob: Array con las probabilidades de cada predicción
    :param pred: Array con las predicciones generadas por el modelo
    :param numClasses: número de clases
    :param numThreads: número de procesos paralelos
    :param kargs:
    :return:
    """
    # Obtener señales desde predicciones
    if prob.shape[0] == 0:
        return pd.Series()
    # 1) Generar señales desde una clasificación multinómica
    # (one-vs-rest, OvR)
    signal0 = (prob - 1. / numClasses) / (prob * (1. - prob))**.5  # t-values or OvR
    signal0 = pred * (2 * norm.cdf(signal0) - 1)  # signal = side * size
    if "side" in events:
        signal0 *= events.loc[signal0.index, "side"]  # Meta-Etiquetado

    # 2) Computar señal promedio de las operaciones abierta actualmente
    df0 = signal0.to_frame("signal").join(events[["t1"]], how="left")
    df0 = avgActiveSignals(df0, numThreads)
    signal1 = discreteSignal(signal0=df0, stepSize=stepSize)
    return signal1


def avgActiveSignals(signals, numThreads):
    """

    :param signals: Pandas DataFrame con las señales
    :param numThreads:
    :return:
    """
    # Computar la señal promedio de aquellas activas
    # 1) Puntos de tiempo donde las señales cambian (sea que una empiece o termine)
    tPnts = set(signals["t1"].dropna().values)
    tPnts = tPnts.union(signals.index.values)
    tPnts = list(tPnts)
    tPnts.sort()
    out = mpPandasObj(mpAvgActiveSignals, ("molecule", tPnts), numThreads, signals=signals)
    return out


def mpAvgActiveSignals(signals, molecule):
    """
    Al tiempo loc, promedia las señales que todavía estén activas.
    Una señal está activa si:
        a) fue emitida antes o justo en loc
        b) loc ocurre antes del tiempo final de la señal, o el tiempo final es todavía desconocido (NaT)
    :param signals: Pandas DataFrame con las señales
    :param molecule:
    :return:
    """
    out = pd.Series()
    for loc in molecule:
        df0 = (signals.index.values <= loc) & ((loc < signals['t1']) | pd.isnull(signals['t1']))
        act = signals[df0].index
        if len(act) > 0:
            out[loc] = signals.loc[act, "signal"].mean()
        else:
            out[loc] = 0  # Sin señales activas a este momento.
    return out


def discreteSignal(signal0, stepSize):
    """
    Discretiza la señal
    :param signal0:
    :param stepSize: paso de discretización
    :return:
    """
    signal1 = (signal0 / stepSize).round() * stepSize
    signal1[signal1 > 1] = 1  # Techo
    signal1[signal1 > -1] = -1  # Piso
    return signal1


def betSize(w, x):
    """
    Calcula tamaño de apuesta
    :param w: coeficiente que regula el ancho de la función sigmoide
    :param x: divergencia entre el precio del mercado actual y la predicción
    :return: tamaño de la apuesta
    """
    return x * (w + x**2)**-.5


def getTPos(w, f, mP, maxPos):
    """
    Tamaño de la posición objetivo asociado a la predicción f
    :param w: coeficiente que regula el ancho de la función sigmoide
    :param f: precio predicho
    :param mP: precio del mercado
    :param maxPos: máxima posición
    :return: tamaño de la posición objetivo
    """
    return int(betSize(w, f - mP) * maxPos)


def invPrice(f, w, m):
    """
    Función inversa del tamaño de la apuesta.
    :param f: precio predicho.
    :param w: coeficiente que regula el ancho de la función sigmoide.
    :param m: j / Q
    :return: resultado de la función inversa del tamaño de la posición.
    """
    return f - m * (w / (1 - m**2))**.5


def limitPrice(tPos, pos, f, w, maxPos):
    """
    Calcula el precio límite para el caso de breakevent
    :param tPos: tamaño de posición objetivo asociado a la predicción f
    :param pos: tamaño de posición actual
    :param f: precio predicho
    :param w: coeficiente que regula el ancho de la función sigmoide
    :param maxPos: máximo tamaño de posición
    :return: precio de breakeven
    """
    sgn = (1 if tPos >= pos else -1)
    lP = 0
    for j in range(abs(pos + sgn), abs(tPos + 1)):
        lP += invPrice(f, w, j / float(maxPos))
    lP /= tPos - pos
    return lP


def getW(x, m):
    # 0 < alpha < 1
    return x**2 * (m**-2 - 1)


def main():
    pos = 0         # posición actual
    maxPos = 100    # posición máxima
    mP = 100        # precio del mercado
    f = 115         # precio predicho
    wParams = {'divergence': 10, 'm': .95}
    # Calibramos w
    w = getW(wParams["divergence"], wParams["m"])
    # Obtener Posición a tiempo t
    tPos = getTPos(w, f, mP, maxPos)
    # Precio límite para la orden
    lP = limitPrice(tPos, pos, f, w, maxPos)
    return


if __name__ == '__main__':
    main()
