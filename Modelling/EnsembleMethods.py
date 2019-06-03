from scipy.misc import comb


def bagging_accuracy(N, p, k):
    """
    Probabilidad para condición necesaria, aunque no suficiente, para que
    se aumente la exactitud del estimador bagging.
    :param N: Número de estimadores
    :param p: la probabilidad de hacer una predicción correcta
    :param k: número de clases
    :return:
    """
    p_ = 0
    for i in range(0, int(N / k) + 1):
        p_ += comb(N, i) * p ** i * (1 - p) ** (N - i)
    return p_

def necesaries_estimators(p, k):
    """
    Cálcula el número de estimadores necesarios como mínimo para
    aumentar la exactitud del estimador
    :param p: probabilidad de hacer una predicción correcta
    :param k: número de clases
    :return: número de estimadores
    """
    N = p * (p - 1 / k) ** (-2)

    return N