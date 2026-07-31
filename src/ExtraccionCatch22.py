import os

import numpy as np
import pandas as pd
from aeon.transformations.collection.feature_based import Catch22
from aeon.transformations.collection.feature_based._catch22 import feature_names

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DATA_DIR = os.path.join(_ROOT, 'data')
_RESULTS_DIR = os.path.join(_ROOT, 'results')

SERIES = [
    'Total_Consistent',
    'Via_Aerea',
    'Via_Terrestre',
    'Via_Maritima',
    'Pais_El Salvador',
    'Pais_Guatemala',
    'Pais_Estados Unidos de América',
]


def CargarSeries():
    ruta = os.path.join(_DATA_DIR, 'series_de_tiempo_completas.csv')
    return pd.read_csv(ruta, index_col='Fecha', parse_dates=True).asfreq('MS')


def Limpiar(serie):
    return serie.interpolate(method='time').fillna(0).clip(lower=0)


def Normalizar(valores):
    return (valores - valores.mean()) / valores.std()


def ExtraerCaracteristicas(df):
    valores = np.array([Normalizar(Limpiar(df[c]).values) for c in SERIES])
    transformador = Catch22(outlier_norm=False, replace_nans=True)
    matriz = transformador.fit_transform(valores.reshape(len(SERIES), 1, -1))
    return pd.DataFrame(matriz, index=SERIES, columns=feature_names)


def main():
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    caracteristicas = ExtraerCaracteristicas(CargarSeries())
    salida = os.path.join(_RESULTS_DIR, 'Catch22Features.csv')
    caracteristicas.to_csv(salida, index_label='Serie')

    print(f'Series procesadas: {len(caracteristicas)}')
    print(f'Caracteristicas por serie: {caracteristicas.shape[1]}')
    print(f'Valores faltantes: {int(caracteristicas.isna().sum().sum())}')
    print()
    print(caracteristicas.round(4).to_string())


if __name__ == '__main__':
    main()
