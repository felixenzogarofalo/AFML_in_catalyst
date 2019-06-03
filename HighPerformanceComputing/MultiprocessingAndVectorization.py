import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import datetime as dt
import sys

def mpPandasObj(func, pdObj, numThreads=24, mpBatches=1, linMols=True, **kargs):
    """
    Paraleliza trabajos, retorna un DataFrame o Series de pandas
    :param func: función a ser paralelizada. Devuelve un Dataframe
    :param pdObj[0]: Nombre de argumento utilizado para pasar la molecula
    :param pdObj[1]: Lista de átomos que serán agrupados en moléculas
    :param numThreads:
    :param mpBatches:
    :param linMols:
    :param kargs: Cualquier otro argumento necesario para la función
    :return: DataFrame o Series de pandas

    Ejemplo: df1 = mpPandasObj(func,("molecule", df0.index), 24, **kargs)
    """
    if linMols:
        parts = linParts(len(pdObj[1]), numThreads * mpBatches)
    else:
        parts = nestedParts(len(pdObj[1]), numThreads * mpBatches)
    jobs = []
    for i in range(1, len(parts)):
        job = {pdObj[0]: pdObj[1][parts[i - 1]:parts[i]], "func": func}
        job.update(kargs)
        jobs.append(job)
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads=numThreads)
    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out
    for i in out:
        df0 = df0.append(i)
    df0 = df0.sort_index()
    return df0



def linParts(numAtoms, numThreads):
    # Partición de datos con un mismo bucle
    parts = np.linspace(0, numAtoms, min(numThreads, numAtoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts

def nestedParts(numAtoms, numThreads, upperTriang=True):
    # Partición de átomos con un bucle interno
    parts = [0]
    numThreads_ = min(numThreads, numAtoms)
    for num in range(numThreads_):
        part = 1 + 4 * (parts[-1]**2 + parts[-1] + numAtoms * (numAtoms + 1.0) / numThreads_)
        part = (-1 + part**0.5) / 2.0
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upperTriang:
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts

def processJobs_(jobs):
    # Correr trabajos secuancialmente, para depurar
    out = []
    for job in jobs:
        out_ = expandCall(job)
        out.append(out_)
        return out

def expandCall(kargs):
    # Expande los argumentos de un función callback, kargs["func"]
    func = kargs["func"]
    del kargs["func"]
    out = func(**kargs)
    return out


def reportProgress(jobNum,numJobs,time0,task):
    # Report progress as asynch jobs are completed
    msg = [float(jobNum) / numJobs, (time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = timeStamp+' '+str(round(msg[0]*100,2))+'% ' + task + ' done after '+ \
          str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2],2)) + ' minutes.'
    if jobNum<numJobs:
        sys.stderr.write(msg+' \ r')
    else:
        sys.stderr.write(msg+' \ n')
    return


def processJobs(jobs, task=None, numThreads=24):
    # Run in parallel.
    # jobs must contain a ’func’ callback, for expandCall
    if task is None:
        task = jobs[0]['func'].__name__
    pool = mp.Pool(processes=numThreads)
    outputs,out,time0 = pool.imap_unordered(expandCall,jobs),[], time.time()
    # Process asynchronous output, report progress
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task)
    pool.close()
    pool.join()  # this is needed to prevent memory leaks
    return out

