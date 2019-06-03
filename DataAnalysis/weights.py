import numpy as np
import pandas as pd
from SupervisedLearningIntraday.HighPerformanceComputing.MultiprocessingAndVectorization import processJobs, processJobs_

def mpNumCoEvents(closeIdx, t1, molecule):
    """
    Computa el número de eventos concurrentes por barra.
    :param closeIdx:
    :param t1:
    :param molecule:
    :return:
    """
    # 1) Hallar eventos que se solapen en el período de la molécula
    t1 = t1.fillna(closeIdx[-1])  # Eventos aun no terminados deben impactar otros pesos
    t1 = t1[t1 > molecule[0]]  # Eventos que terminan en la molécula o después de esta
    t1 = t1.loc[:t1[molecule].max()]  # Eventos que inician en o antes de t1[molecule].max()
    # 2) Contar los eventos que se solapan en una barra
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1] + 1])
    for tIn, tOut in t1.iteritems(): count.loc[tIn:tOut] += 1
    return count.loc[molecule[0]:t1[molecule].max()]


def mpSampleTW(t1, numCoEvents, molecule):
    """
    Derivar el promedio de unicidad sobre el lapso de vida del elemento
    :param t1:
    :param numCoEvents:
    :param molecule:
    :return:
    """
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (1.0 / numCoEvents.loc[tIn:tOut]).mean()
    return wght


def getIndMatrix(barIx, t1):
    """
    Obtener la matrix indicadora
    :param barIx:
    :param t1:
    :return:
    """
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0, t1) in enumerate(t1.iteritems()):
        indM.loc[t0:t1, i] = 1.0
    return indM


def getAvgUniqueness(indM):
    # Promediar unicidad desde la matriz indicadora
    c = indM.sum(axis=1)  # Concurrencia
    u = indM.div(c, axis=0)  # Unicidad
    avgU = u[u > 0].mean()  # Unicidad promedio
    return avgU


def seqBootstrap(indM,sLength = None):
    # Generate a sample via sequential bootstrap
    if sLength is None:sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series()
        for i in indM:
            indM_ = indM[phi + [i]] # reduce indM
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU/avgU.sum() # draw prob
        phi += [np.random.choice(indM.columns, p=prob)]
    return phi


def getRndT1(numObs, numBars, maxH):
    # Serie t1 aleatoria
    t1 = pd.Series()
    for i in range(numObs):
        ix = np.random.randint(0, numBars)
        val = ix + np.random.randint(1, maxH)
        t1.loc[ix] = val
    return t1.sort_index()


def auxMC(numObs,numBars,maxH):
    # Parallelized auxiliary function
    t1 = getRndT1(numObs,numBars,maxH)
    barIx = range(t1.max()+1)
    indM = getIndMatrix(barIx,t1)
    phi = np.random.choice(indM.columns,size = indM.shape[1])
    stdU = getAvgUniqueness(indM[phi]).mean()
    phi = seqBootstrap(indM)
    seqU = getAvgUniqueness(indM[phi]).mean()
    return { 'stdU':stdU,'seqU':seqU }


def mainMC(numbObs=10, numBars=100, maxH=5, numIters=1000, numThreads=24):
    # Experimentos Monte Carlo
    jobs = []
    for i in range(int(numIters)):
        job = {"func": auxMC,
               "numObs": numbObs,
               "numBars": numBars,
               "maxH": maxH}
        jobs.append(job)
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads=numThreads)
    print(pd.DataFrame(out).describe())
    return


def mpSampleW(t1, numCoEvents, close, molecule):
    # Derivar el peso de ejemplar por la atribución devuelta
    ret = np.log(close).diff()
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (ret.loc[tIn:tOut] / numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()


def getTimeDecay(tW, clfLastW=1.0):
    # Aplicar un decaimiento lineal por pieza a la unicidad obserbada tW
    # Las observaciones más recientes obtienen un weight=1, las más antiguas obtiene weight=clfLastW
    clfW = tW.sort_index().cumsum()
    if clfLastW >= 0:
        slope = (1.0 - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1.0 / ((clfLastW + 1) * clfW.iloc[-1])
    const = 1.0 - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0
    print(const, slope)
    return clfW


if __name__ == '__main__':
    t1 = pd.Series([2, 3, 5], index=[0, 2, 4])
    print(t1)
    barIx = range(t1.max() + 1)
    print(barIx)
    indM = getIndMatrix(barIx, t1)
    print(indM)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    print(phi)
    print("Unicidad estándar:", getAvgUniqueness(indM[phi]).mean())
    phi = seqBootstrap(indM)
    print(phi)
    print("Unicidad secuencial:", getAvgUniqueness(indM[phi]).mean())
    mainMC()