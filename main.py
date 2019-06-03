import matplotlib.pyplot as plt
from SupervisedLearningIntraday.synthetic_data import SyntheticData
from SupervisedLearningIntraday.features import create_up_down_dataframe as updown
from SupervisedLearningIntraday.features import bbands, get_down_cross, get_up_cross
from SupervisedLearningIntraday.features import stochasticOsillator as stoch
from SupervisedLearningIntraday.features import dobleMomentumStrategy as dms
import numpy as np
import pandas as pd
from logbook import Logger
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier)
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os

from catalyst import run_algorithm
from catalyst.api import (record, symbol, order_target_percent, order_percent, )
from catalyst.api import order, record, symbol, symbols

import SupervisedLearningIntraday.DataAnalysis.FinantialDataStructure as fds
import SupervisedLearningIntraday.DataAnalysis.labelling as labelling
import SupervisedLearningIntraday.DataAnalysis.weights as weights
import SupervisedLearningIntraday.DataAnalysis.FractionallyDifferentiatedFeatures as fdf
import SupervisedLearningIntraday.HighPerformanceComputing.MultiprocessingAndVectorization as mpv
from SupervisedLearningIntraday.DataAnalysis.weights import mpNumCoEvents, mpSampleW
from SupervisedLearningIntraday.Modelling.CrossValidation import cvScore
from SupervisedLearningIntraday.Modelling.HyperparameterTuning import clfHyperFit
from SupervisedLearningIntraday.Modelling.HyperparameterTuning import logUniform
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
import multiprocessing
import time

NAMESPACE = 'Supervised Machine Learning'
log = Logger(NAMESPACE)


def initialize(context):
    # Lista de activos
    context.assets = ["btc_usdt"]
    context.symbols = symbols("btc_usdt")

    # Definir carácteristicas
    context.row_features = ["open", "high", "low", "close", "volume"]
    context.features = ["open", "high", "low", "close", "volume"]

    context.initial_value = context.portfolio.starting_cash

    context.i = 0
    context.bar_period = 15

    context.random_state = 42
    context.n_estimators = 1000  # 400
    context.max_deep = 10
    context.n_jobs = 1
    context.lookback_w = 30
    context.lookforward_w = 5
    context.model_trained = False
    context.training_data_size = 50000 # 700000
    context.model_pickle_directory = "/home/enzo/PycharmProjects/SupervisedLearningIntraday/TrainedModels/"


    if os.path.isfile(context.model_pickle_directory + "meta_model.pkl"):
        context.model = joblib.load(context.model_pickle_directory + "model.pkl")
        context.meta_model = joblib.load(context.model_pickle_directory + "meta_model.pkl")
        context.model_trained = True
        print("Modelo Pre-Entrenado")
    context.lags = 5

    context.invested = False
    context.cur_prices = np.zeros(context.lags + 1)
    context.cur_returns = np.zeros(context.lags)

    # Advanced parameters
    context.minRet = 0.01
    context.cpus = multiprocessing.cpu_count()
    context.vol_delta = pd.Timedelta(hours=1)
    context.vb_delta = pd.Timedelta(hours=1)
    context.vol_freq = 10  # 10
    context.numThreads = 24
    context.d = 0.25  # este es el factor fraccional de Fractional Differentieted Features
                      # y debe ser verfificado en el ambiente de investigación para
                      # determinar su magnitud utilizando la función plotMinFFD()


def handle_data(context, data):
    if context.i % context.bar_period == 0:
        if not context.model_trained:
            print("SML INFO: Entrenando el modelo '" + NAMESPACE + "'")
            train_model(context, data)
            context.model_trained = True
            print("SML INFO: Modelo entrenado")

        h1 = data.history(context.symbols,
                          context.row_features,
                          bar_count=2000,
                          frequency="1T",
                          )

        h1 = h1.swapaxes(2, 0)

        # Por ahora se está trabajando con un solo activo
        h1 = h1.iloc[0]
        df = fds.VolumeBarSeries(df=h1, vol_frequency=context.vol_freq).process_data()

        # Primero calculamos los estocásticos mayor y menor
        features = stoch(df, lookback=14, softened=3, name="minor")
        features = stoch(features, lookback=56, softened=3, name="mayor")
        # Luego calculamos las señales de la estrategia de doble-momentum
        # La función dms hace esto agregando una nueva columna llamada "DMStrat"
        features = dms(features, mayor="mayor", minor="minor", threshold=0.2)

        # Parámetros de Bandas Bollinger
        ave = df.close.rolling(20).mean()
        sd = df.close.rolling(20).std(ddof=0)
        upper_band = ave + (sd * 2)
        down_band = ave - (sd * 2)
        up_crit1 = df.close.shift(1).iloc[-1] < upper_band.shift(1).iloc[-1]
        up_crit2 = df.close.iloc[-1] > upper_band.iloc[-1]

        if up_crit1 and up_crit2:
            up_cross = True
        else:
            up_cross = False

        down_crit1 = df.close.shift(1).iloc[-1] > down_band.shift(1).iloc[-1]
        down_crit2 = df.close.iloc[-1] < down_band.iloc[-1]

        if down_crit1 and down_crit2:
            down_cross = True
        else:
            down_cross = False

        FDFeatures = fdf.fracDiff(features[["close", "volume", "minorD", "mayorD", "DMStrat"]], context.d, thres=0.01)
        inputs = FDFeatures.iloc[-1]
        signal = inputs["DMStrat"]

        if up_cross or down_cross:
            direction = context.model.predict(inputs.values.reshape(1, -1))
            confirmation_input = [[FDFeatures.close.iloc[-1], direction]]
            confirmation = context.meta_model.predict(confirmation_input)

            if up_cross and confirmation == 1:
                size = context.meta_model.predict_proba(confirmation_input)[0][2]
                for i, asset in enumerate(context.symbols):
                    order_target_percent(asset, size)
                    print("Compra: ", size)
            elif down_cross and confirmation == 1:
                size = context.meta_model.predict_proba(confirmation_input)[0][2]
                for i, asset in enumerate(context.symbols):
                    order_target_percent(asset, -size)
                    print("Venta: ", size)

        record(btc=data.current(symbol(context.assets[0]), "price"))
        print(str("[" + data.current_dt.strftime("%x")),
              str(data.current_dt.strftime("%X") + "]"),
              "SML INFO:",
              "Valor de Portafolio:", "%.2f" % context.portfolio.portfolio_value)

    context.i += 1


def analyze(context, perf):
    # Plot the portfolio and asset data.
    ax1 = plt.subplot(211)
    perf[['portfolio_value']].plot(ax=ax1)
    ax1.set_ylabel('Portfolio\nValue\n(USD)')
    ax2 = plt.subplot(212, sharex=ax1)
    perf.btc.plot(ax=ax2)
    ax2.set_ylabel('Precio Bitcoin\n(USD)')
    plt.show()


def get_data(context, data_, window):
    # Crear ventana de datos.
    h1 = data_.history(context.symbols,
                       context.row_features,
                       bar_count=window,
                       frequency="1T",
                       )

    h1 = h1.swapaxes(2, 0)

    # Por ahora se está trabajando con un solo activo
    data = h1.iloc[0]

    features, label = updown(data=data,
                             lookback_w=context.lookback_w,
                             lookforward_w=context.lookforward_w,
                             up_down_factor=2.0,
                             percent_factor=0.01)

    return features

# ===========================
#   Model Training
# ===========================


def train_model(context, _data):
    # Crear clase de datos sintéticos

    # Crear ventana de datos.
    h1 = _data.history(context.symbols,
                       context.row_features,
                       bar_count=context.training_data_size,
                       frequency="1T",
                       )

    h1 = h1.swapaxes(2, 0)

    # Por ahora se está trabajando con un solo activo
    data = h1.iloc[0]

    print("Datos en bruto:", data.shape)
    print("Inicio de conjunto de enrenamiento: ", data.index[0])
    print("Fin de conjunto de enrenamiento: ", data.index[-1])

    ##################################
    # Aplicar scripts de "Advances in Financial Machine Learning"
    ##################################

    # Crear Barras de Volumen
    vbs = fds.VolumeBarSeries(df=data, vol_frequency=context.vol_freq).process_data()

    print("Barras de volumen:", vbs.shape)

    # Método de la Triple-Barrera
    """
    Para crear la lista de eventos vamos a utilizar una estrategía de reversión de media basada
        en Bandas Bollinger.
    En esta etapa el modelo solo sugiere el lado de la operación, pero no la cantidad a invertir.
    """
    window = 20
    bb_df = pd.DataFrame()
    bb_df["price"], bb_df["ave"], bb_df["upper"], bb_df["lower"] = bbands(vbs.close, window=window, numsd=2)
    bb_df.dropna(inplace=True)

    bb_down = get_down_cross(bb_df, "price")
    bb_up = get_up_cross(bb_df, "price")

    bb_side_up = pd.Series(-1, index=bb_up.index)
    bb_side_down = pd.Series(1, index=bb_down.index)
    bb_side_raw = pd.concat([bb_side_up, bb_side_down]).sort_index()

    # Agregar características extra de entrenamiento
    # Primero calculamos los estocásticos mayor y menor
    features = stoch(vbs, lookback=14, softened=3, name="minor")
    features = stoch(features, lookback=56, softened=3, name="mayor")
    # Luego calculamos las señales de la estrategia de doble-momentum
    # La función dms hace esto agregando una nueva columna llamada "DMStrat"
    features = dms(features, mayor="mayor", minor="minor", threshold=0.2)

    tEvents = pd.DatetimeIndex(bb_side_raw.index)
    TpSl = [1, 1]
    vol = labelling.getVol(vbs.close, delta_time=context.vol_delta)
    vertical_barrier = labelling.addVerticalBarrier(tEvents, vbs.close, delta_time=context.vb_delta)

    bb_events = labelling.getEvents(close=vbs.close,
                                    tEvents=tEvents,
                                    TpSl=TpSl,
                                    trgt=vol,
                                    minRet=context.minRet,
                                    numThreads=context.cpus,
                                    t1=vertical_barrier,
                                    side=None)

    bins = labelling.getBins(bb_events, vbs.close)

    # Meta Labeling

    meta_events = labelling.getEvents(close=vbs.close,
                                      tEvents=tEvents,
                                      TpSl=[2, 1],
                                      trgt=vol,
                                      minRet=0.01,
                                      numThreads=24,
                                      t1=vertical_barrier,
                                      side=bins.bin)

    meta_bins = labelling.getBins(meta_events, vbs.close, vb_offset=context.vb_delta)

    labels = pd.concat({"price": vbs.close.loc[bins.index],
                        "volume": vbs.volume.loc[bins.index],
                        "minorD": features.minorD.loc[bins.index],
                        "mayorD": features.mayorD.loc[bins.index],
                        "DMStrat": features.DMStrat.loc[bins.index],
                        "direction": bins.bin,
                        "size": meta_bins.bin}, axis=1)

    # Pesos de observaciones
    # La razón principal de este método es que deseamos ponderar una observación en función de los
    # retornos logarítmicos absolutos que pueden ser atribuidos únicamente a dicha observación.

    # Número de eventos concurrentes
    numCoEvents = mpv.mpPandasObj(weights.mpNumCoEvents,
                                  ('molecule', bb_events.index),
                                  context.numThreads,
                                  closeIdx=vbs.close.index,
                                  t1=bb_events['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(vbs.close.index).fillna(0)

    # tW: promedio de unicidad de una etiqueta
    out = mpv.mpPandasObj(weights.mpSampleTW,
                         ('molecule', bb_events.index),
                         context.numThreads,
                         t1=bb_events['t1'],
                         numCoEvents=numCoEvents)

    out = out.to_frame().rename(index=str, columns={0: "tW"})

    out['w'] = mpv.mpPandasObj(weights.mpSampleW,
                               ('molecule', bb_events.index),
                               context.numThreads,
                               t1=bb_events['t1'],
                               numCoEvents=numCoEvents,
                               close=vbs.close)

    out['w'] *= out.shape[0] / out['w'].sum()

    sample_weight = out["w"]

    # Entrenar los modelos
    # Modelo de dirección

    # Pre-procesamiento de las características
    """
    Método de las características diferenciadas de formar fracccional. (FFD)
    
    Esta es una forma de de transformar series temporales no-estacionarias, que tienen cierto grado de correlación, 
    en series temporales estacionarias, mientras que se mantiene la correlación de los datos. Esto es una ventaja pues, 
    si bien la estacionariedad en una condición necesaria para hacer estimaciónes sobre conjuntos de datos nuevos, 
    muchos de los métodos utilizados para lograr dicha estacionariedad implican la pérdida de memoria (correlación 
    temporar de los datos de la serie) lo que a su vez implica perder poder predictivo. Con este método se logra 
    obterner estacionariedad mientras se preserva el máximo de memoría.
    """
    # Vamos a aplicar FFD a las características de entrada
    FDFeatures = fdf.fracDiff(labels[["price", "volume", "minorD", "mayorD", "DMStrat"]], context.d, thres=0.01)
    # Convertir el índice de los pesos de ejemplar en datetime
    sample_weight.index = pd.to_datetime(sample_weight.index)
    sample_weight = sample_weight.loc[FDFeatures.index]

    X = FDFeatures[["price", "volume", "minorD", "mayorD", "DMStrat"]]  # .reshape(-1, 1)
    y = labels["direction"].loc[FDFeatures.index]

    # Configuración del Bosque Aleatorio y Agregación Bootstrap para evitar el sobre ajuste, utilizando la biblioteca
    # de sklearn

    barIx = FDFeatures.index
    indM = weights.getIndMatrix(barIx=barIx,
                                t1=vertical_barrier)

    phi = np.random.choice(indM.columns, size=indM.shape[1])
    avgU = weights.getAvgUniqueness(indM[phi]).mean()

    context.model = RandomForestClassifier(n_estimators=1,
                                           criterion='entropy',
                                           bootstrap=False,
                                           class_weight='balanced_subsample')

    context.model = BaggingClassifier(base_estimator=context.model,
                                      n_estimators=context.n_estimators,
                                      max_samples=avgU,
                                      max_features=1.)

    start_all = time.time()
    start = time.time()

    """
    print("SML INFO: Ajustando Hiper-Parámetros de modelo primario.")
    # Antes de pasar a realizar el entrenamiento del modelo utilizando Validación cruzada, es necesario ajustar
    # los hiper-parámetros del modelo. Haremos estos utilizando la función "clfHyperFit"

    # Creemos un diccionario de hiper-parámetros y sus valores para crear la búsqueda por rejilla.
    # Nota: debe tomarse en cuenta el tipo de función que define el comportamiento del parámetro a optimizar.
    # Puede que la función de dichos parámetros no sea lineal y por ende una distribución lineal uniforme de los
    # valores a explorar, resultaría ineficiente. En estos casos podría utilizarse, más bien, una distribución
    # logarítmica de los valores. La función "logUniform" implementa este tipo de distribución.
    # Ejemplo:
    # a = 1e-3
    # b = 1e3
    # size = 1000
    # vals = logUniform(a=a, b=b).rvs(size=size)

    param_grid = {'RandomForestClassifier__n_estimators': [1, 5, 10, 100, 1000],
                  'RandomForestClassifier__max_features': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

    # Debemos crear un Pipeline con los pasos donde queremos optimizar los parámetros.
    # pipe_clf = make_pipeline(RandomForestClassifier())
    pipe_clf = Pipeline([('RandomForestClassifier', RandomForestClassifier())])

    # La variable "bagging" acá representa 3 parámetro a ser pasados en BaggingClassifier:
    # 1. n_estimators, 2. max_samples y 3. max_features
    # Si el segundo valor es cero no se aplica el bagging
    bagging = [context.n_estimators, 0, 1.]

    fit_params = {"RandomForestClassifier__eval_set": [(X, y)],
                  "RandomForestClassifier__sample_weight": sample_weight}

    context.model = clfHyperFit(feat=X,
                                lbl=y,
                                t1=vertical_barrier.loc[FDFeatures["price"].index],
                                pipe_clf=pipe_clf,
                                param_grid=param_grid,
                                cv=3,
                                bagging=bagging,
                                rndSearchIter=1,
                                n_jobs=-1,
                                pctEmbargo=0,
                                sample_weight=sample_weight)
    """

    print("SML INFO: Entrenando modelo primario.")

    score = cvScore(clf=context.model,
                    X=X,
                    y=y,
                    sample_weight=sample_weight,
                    scoring='accuracy',  # 'neg_log_loss',
                    t1=vertical_barrier.loc[FDFeatures["price"].index],
                    cv=8,  # número de divisiones para el Cross-Validation
                    cvGen=None,  # Generador de Cross-Validator, si existiera previamente.
                    pctEmbargo=0.01
                    )

    print("Puntaje de Cross-Validation para modelo primario: ", score)
    end = time.time()
    print("Tiempo de entrenamiento de modelo 01: ", str(end - start))

    # Modelo de tamaño - Meta-modelo
    X_m_fdf = pd.concat({"price": FDFeatures.price,
                         "direction": labels.direction.loc[FDFeatures.index]}, axis=1)

    X_m = X_m_fdf[["price", "direction"]].fillna(method="bfill")
    y_m = labels["size"].loc[FDFeatures.index].fillna(method="bfill")

    context.meta_model = RandomForestClassifier(n_estimators=1,
                                                criterion='entropy',
                                                bootstrap=False,
                                                class_weight='balanced_subsample')

    context.meta_model = BaggingClassifier(base_estimator=context.meta_model,
                                           n_estimators=context.n_estimators,
                                           max_samples=avgU,
                                           max_features=1.)
    start = time.time()

    print(labels["size"].loc[FDFeatures.index].fillna(method="bfill"))

    """
    print("SML INFO: Ajustando Hiper-Parámetros de modelo secundario.")
    
    context.meta_model = clfHyperFit(feat=X,
                                     lbl=y,
                                     t1=vertical_barrier.loc[FDFeatures["price"].index],
                                     pipe_clf=pipe_clf,
                                     param_grid=param_grid,
                                     cv=3,
                                     bagging=bagging,
                                     rndSearchIter=0,
                                     n_jobs=-1,
                                     pctEmbargo=0,
                                     sample_weight=sample_weight,
                                     fit_params={})
    """

    print("SML INFO: Entrenando modelo secundario.")

    meta_score = cvScore(clf=context.meta_model,
                         X=X_m_fdf.fillna(method="bfill"),
                         y=labels["size"].loc[FDFeatures.index].fillna(method="bfill"),
                         sample_weight=sample_weight,
                         scoring='accuracy',  # 'neg_log_loss',
                         t1=vertical_barrier.loc[FDFeatures["price"].index],
                         cv=8,  # número de divisiones para el Cross-Validation
                         cvGen=None,  # Generador de Cross-Validator, si existiera previamente.
                         pctEmbargo=0.01
                         )

    print("Puntaje de Cross-Validation para modelo secundario: ", meta_score)

    print("Pickling model...")
    model_name = "ml_model_rf_s" + str(X.shape[0]) + "_p" + str(context.bar_period) + ".pkl"
    joblib.dump(context.model, context.model_pickle_directory + "model.pkl")
    joblib.dump(context.meta_model, context.model_pickle_directory + "meta_model.pkl")

    context.model_trained = True


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
        start=pd.to_datetime('2018-02-1', utc=True),
        end=pd.to_datetime('2018-02-28', utc=True),
    )
