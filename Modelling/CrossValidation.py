import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold
from sklearn.metrics import log_loss, accuracy_score

def getTrainTimes(t1, testTimes):
    """
    Dados los tiempos testTimes, hallar los tiempos de las observaciones de entrenamiento.
    :param t1: - t1.index -> tiempo en que la observación comienza
               - t1.value -> tiempo en que la observación termina
    :param testTimes: tiempos de las obsrevaciones de prueba, es decir, es el t1 del test
    :return:
    """
    trn = t1.copy(deep=True)
    for i, j in testTimes.iteritems():
        # Cuando el entrenamiento empieza dentro del set de prueba
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index
        # Cuando el entrenamiento finaliza dentro del set de prueba
        df1 = trn[(i <= trn) & (trn <= j)].index
        # Cuando el set de entrenamiento envuelve al set de prueba
        df2 = trn[(trn.index <= i) & (j <= trn)].index
        trn = trn.drop(df0.union(df1).union(df2))
    return trn

def getEmbargoTimes(times, pctEmbargo=0.01):
    # Obtener el tiempo del embargo para cada barra
    step = int(times.shape[0] * pctEmbargo)
    if step == 0:
        mbrg = pd.Series(times, index=times)
    else:
        mbrg = pd.Series(times[step:], index=times[:-step])
        mbrg = mbrg.append(pd.Series(times[-1], index=times[-step]))
    return mbrg

class PurgedKFold(_BaseKFold):
    """
    Extiende la clase Kfold para trabajar con etiquetas que espacie los intervalos
    Los datos de entrenamiento son purgados de observaciones que solapen el intervalo de las
        etiquetas de prueba.
    El conjunto de entrenamiento se asume contiguo (shuffle=False), sin muestras de entrenamiento
        en medio.
    """
    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.01):
        if not isinstance(t1, pd.Series):
            raise ValueError("Label Through Dates must be a pd.Series")
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and ThruDateValues must have he same index")
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i, j in test_starts:
            # Inicio del conjunto de datos
            t0 = self.t1.index[i]
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            # El entrenamiento correcto, con embargo
            if maxT1Idx < X.shape[0]:
                train_indices=np.concatenate((train_indices, indices[maxT1Idx + mbrg:]))
            yield train_indices, test_indices


def cvScore(clf, X, y, sample_weight, scoring='neg_log_loss', t1=None, cv=None, cvGen=None, pctEmbargo=None):
    """

    :param clf: modelo clasificador
    :param X: DataFrame pandas con características de entrenamiento
    :param y: DataFrame pandas con etiquetas de entrenamiento
    :param sample_weight: Numpy Array con pesos de unicidad
    :param scoring: String identificando el método para calcular el puntaje de la Validación Cruzada
    :param t1: tiempo de inicio de los eventos
    :param cv: número de divisiones para la Validación Cruzada
    :param cvGen: Generador si existiera alguna previamente
    :param pctEmbargo: procentaje de separación entre los grupos de entrenamiento y prueba
    :return:
    """
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')
    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv,
                            t1=t1,
                            pctEmbargo=pctEmbargo)  # purged
    score = []
    for train, test in cvGen.split(X=X):

        fit = clf.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight.iloc[train].values)

        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(y.iloc[test],
                               prob,
                               sample_weight=sample_weight.iloc[test].values,
                               labels=clf.classes_)
        else:
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(y.iloc[test],
                                    pred,
                                    sample_weight=sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)
