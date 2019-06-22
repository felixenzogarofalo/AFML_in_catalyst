import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous, kstest
from Modelling.CrossValidation import PurgedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline


class MyPipeline(Pipeline):
    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0]+'__sample_weight'] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)

    """
    ____________________________________________________________________________________________________________________
    
    El siguiente código es un ejemplo de como implementar el Pipeline sobre un clasificador 
    basado en Support Vector Machine.
    param_grid = {'svc__C': [1e-2, 1e-1, 1, 10, 100],
                  'svc__gamma': [1e-2, 1e-1, 1, 10, 100]}
    scoring_func = 'accuracy'

    pipe_svc = make_pipeline(StandardScaler(),
                             SVC(random_state=RANDOM_STATE))
    ____________________________________________________________________________________________________________________
    
    El siguiente código es una implementación del Pipeline sobre un clasificador basado en Bosques aleatorios.
    rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator, random_state=0)

    rt_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
    pipeline = make_pipeline(rt, rt_lm)
    ____________________________________________________________________________________________________________________
    """


def clfHyperFit(feat, lbl, t1, pipe_clf, param_grid, cv=3, bagging=[0, None, 1.], rndSearchIter=0, n_jobs=-1,
                pctEmbargo=0, **fit_params):
    """
    Esta función implementa a su vez la función pre-construida de sklearn llamada "GridSearchCV", que conduce una
    búsqueda exhaustiva para combinación de parámetros que maximicen a su ves el rendimiento de la
    Validación Cruzada (Cross-Validation) de acuerdo a una función puntaje definida por el usuario.
    Nota: acá se utiliza la función personalizada "PurgedKFold" que toma en cuenta en "embargo" y la unicidad
    a la hora de crear los conjuntos de entrenamiento-prueba para realizar la Validación Cruzada.
    :param feat: dato tipo matriz (pandas.DataFrame) con forma = [n_muestras, n_características]
    :param lbl: dato tipo matriz (pandas.Series) con forma = [n_muestras] o [n_muestras, n_salidas]
                    en caso de más de una etiqueta.
    :param t1: Pandas DataFrame con tiempos de inicio y fin de cada evento
    :param pipe_clf: Pipeline con los pasos a seguir para ajustar el clasificador
    :param param_grid: rejilla de parámetros
    :param cv: número de divisiones para la validación cruzada.
    :param bagging:
    :param rndSearchIter: si es cero, no se toman los parámetros desde una distribución probabilística.
    :param n_jobs: número de trabajos paralelos
    :param pctEmbargo: porcentaje para el "embargo". 1% suele ser suficiente.
    :param sample_weight: pesos de muestras según su unicidad.
    :param fit_params: puede utilizarse para pasar "sample_weight"
    :return:
    """

    """
    Cuando ciertos valores se repiten muchas veces, la mejor manera de medir el rendimiento es con F1-Score, pues
    toma en cuenta la precisión y la exhaustividad. Si se utilizara "accuracy" o "neg_log_loss" podría obtenerse
    un gran puntaje, aun a pesar de que el clasificador no esté aprendiendo realmente a diferenciar las etiquetas
    de las características (por ejemplo, esto ocurre cuando siempre arroja la misma etiqueta sin importar el caso). 
    """
    # El siguiente código asume que "lbl" es un pandas.Series. Si se diera el caso de que existan más de una etiqueta
    # de salida, entonces debería utilizarse pandas.DataFrame. Pero para poder pasarlo a la función set() debe
    # escogerse primero una columna, porque set solo admite datos unidimensionales.
    #
    # Lo que hace el siguiente código es verificar si la etiqueta es binaria, y en caso positivo utiliza
    # Valor F1 para medir el rendimiento.
    if set(lbl.values) == {0, 1}:
        scoring = 'f1'  # F1 score para meta-etiquetado
    else:
        scoring = 'neg_log_loss'  # Simétrico en todos los casos

    # 1) Búsqueda de hiper-parámetros en datos de entrenamiento
    # Primero se busca las divisiones a utilizar en laValidación Cruzada tomando en cuenta el
    # porcentaje de embargo y unicidad.
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged
    # Si "rndSearchIter" == 0 se toman en cuenta todos los parámetros en la búsqueda.
    if rndSearchIter == 0:
        # La variable "gs" viene de "Grid Search" -> Búsqueda por rejilla
        gs = GridSearchCV(estimator=pipe_clf,
                          param_grid=param_grid,
                          scoring=scoring,
                          cv=inner_cv,
                          n_jobs=n_jobs,
                          iid=False,)
    # Si "rndSearchIter" == 1 se toman los parámetros desde una distribución probabilística.
    else:
        gs = RandomizedSearchCV(estimator=pipe_clf,
                                param_distributions=param_grid,
                                scoring=scoring,
                                cv=inner_cv,
                                n_jobs=n_jobs,
                                iid=False,
                                n_iter=rndSearchIter)
    # Notar que en la implementación de fit a continuación no se están pasando lo pesos de la muestras.
    gs = gs.fit(feat, lbl, fit_params["sample_weight"]).best_estimator_


    # 2) Ajustar el modelo validado en todos los datos enteros
    if bagging[1] > 0:
        gs = BaggingClassifier(base_estimator=Pipeline(gs.steps),
                               n_estimators=int(bagging[0]),
                               max_samples=float(bagging[1]),
                               max_features=float(bagging[2]),
                               n_jobs=n_jobs)
        # Acá si se implementa la función fit pasando los pesos de las muestras. Por lo tanto, el Pipeline
        # pasado como parámetro al "BaggingClassifier" debe, a su vez, poder recibir "sample_weight" como parámetro
        # y procesar dicho parámetro correctamente. Esto se logra con la clase "MyPipeline"
        # que hereda de la clase sklearn.pipeline.Pipeline y modifica el método "fit".

        gs = gs.fit(feat, lbl)  # sample_weight=fit_params[gs.base_estimator.steps[-1][0]+'__sample_weight'])
        gs = Pipeline([('bag', gs)])
    return gs


class logUniform_gen(rv_continuous):
    # Números aleatorios distribuidos de forma uniformemente logarítmica entre 1 y e
    def _cdf(self, x):
        return np.log(x / self.a) / np.log(self.b / self.a)


def logUniform(a=1.0, b=np.exp(1)):
    return logUniform_gen(a=a, b=b, name="logUniform")


if __name__ == '__main__':
    a = 1E-3
    b = 1E3
    size = 10000
    vals = logUniform(a=a, b=b).rvs(size=size)
    print(kstest(rvs=np.log(vals), cdf="uniform", args=(np.log(a), np.log(b / a)), N=size))
    print(pd.Series(vals).describe())
    plt.subplot(121)
    pd.Series(np.log(vals)).hist()
    plt.subplot(122)
    pd.Series(vals).hist()
    plt.show()
