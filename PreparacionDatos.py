import pandas as pd

TRAIN_RATIO = 0.7
SERIES = ['Total_Consistent', 'Via_Aerea']


def CargarSeries(ruta='data/series_de_tiempo_completas.csv'):
    return pd.read_csv(ruta, index_col='Fecha', parse_dates=True).asfreq('MS')


def FechaCorte(df):
    fechas = sorted(df.index.unique())
    return pd.Timestamp(fechas[int(len(fechas) * TRAIN_RATIO)])


def DividirSerie(df, columna, corte):
    s = df[columna].interpolate(method='time').fillna(0).clip(lower=0)
    return s.loc[:corte].asfreq('MS'), s.loc[corte:].iloc[1:].asfreq('MS')


def ObtenerConjuntos(ruta='data/series_de_tiempo_completas.csv'):
    df = CargarSeries(ruta)
    corte = FechaCorte(df)
    return {c: DividirSerie(df, c, corte) for c in SERIES}


if __name__ == '__main__':
    for nombre, (train, test) in ObtenerConjuntos().items():
        train.to_csv(f'data/train_{nombre.lower()}.csv', header=[nombre])
        test.to_csv(f'data/test_{nombre.lower()}.csv', header=[nombre])
        print(f'{nombre}: train {len(train)} ({train.index.min():%Y-%m} a '
              f'{train.index.max():%Y-%m}), test {len(test)} '
              f'({test.index.min():%Y-%m} a {test.index.max():%Y-%m})')
