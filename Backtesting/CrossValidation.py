import pandas as pd
import numpy as np
import math


class CombinatorialCV(object):

    def __init__(self, N, k):
        """
        :param N: número de grupos no barajados totales
        :param k: número de grupos de prueba
        """
        self.N = N
        self.k = k
        self.splits_num = 0
        self.paths_num = 0
        self.path_map = []

    def CombinationalSplits(self, groups):
        """
        Crea las divisiones posibles en función del número de grupos totales y de prueba.
        :param groups: lista con la división de grupos, cada elemento de la lista en un Pandas DataFrame
        :return: lista con pares de entrenamiento y prueba.
                    Forma: n_splits x 2 x [train_data_shape or test_data_shape]
                    La segunda dimensión es igual 2: siendo 0 para los datos de entrenamiento
                                                     y 1 para los datos de prueba.
        """
        # Número de posibles divisiones entrenamiento/prueba
        df_i = pd.DataFrame(data={"i": list(range(self.k))})
        numerador = np.prod(self.N - df_i.i.values)
        denominador = math.factorial(self.k)
        self.splits_num = numerador / denominador

        # Número de posibles caminos
        self.paths_num = int(self.k / self.N * self.splits_num)

        splits = []
        split_num = 1
        path_count = [0] * self.paths_num

        # Generación de combinaciones
        for i in range(self.N):
            for j in range(i + 1, self.N):
                testing_groups = []
                training_groups = []
                for k in range(self.N):
                    if (k == i) or (k == j):
                        testing_groups.append(groups[k])
                        path_count[k] += 1
                        self.path_map.append([split_num, k, path_count[k]])
                    else:
                        training_groups.append(groups[k])
                for test_group in testing_groups:
                    yield np.concatenate(training_groups), np.concatenate(testing_groups)
                split_num += 1

    def GroupsSplit(self, df):
        """
        Particiona las T observaciones en N grupos sin barajar
        :param df: Pandas DataFrame conteniendo las T observaciones de características y etiquetas
        :return: lista de grupos
        """
        T = df.shape[0]
        groups = []
        size = int(T / self.N)
        for i in range(1, self.N):
            group = df.iloc[(i - 1) * size: size * i].values
            groups.append(group)
        last_group = df.iloc[(self.N - 1) * size:].values
        groups.append(last_group)
        return groups

