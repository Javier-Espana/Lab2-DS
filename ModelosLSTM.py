import itertools
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from PreparacionDatos import ObtenerConjuntos

SEMILLA = 42
EPOCAS = 500
VALIDACION = 24

CONFIGURACIONES = {
    'LstmSimple': {'capas': [1], 'dropout': [0.0],
                   'ventana': [12, 24], 'unidades': [32, 64], 'lr': [0.01, 0.001]},
    'LstmApilado': {'capas': [2], 'dropout': [0.2],
                    'ventana': [12, 24], 'unidades': [32, 64], 'lr': [0.01, 0.001]},
}


class RedLstm(nn.Module):
    def __init__(self, unidades, capas, dropout):
        super().__init__()
        self.lstm = nn.LSTM(1, unidades, capas, batch_first=True,
                            dropout=dropout if capas > 1 else 0.0)
        self.salida = nn.Linear(unidades, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.salida(h[:, -1])


def CrearSecuencias(valores, ventana):
    x = np.array([valores[i:i + ventana] for i in range(len(valores) - ventana)])
    y = np.array([valores[i + ventana] for i in range(len(valores) - ventana)])
    return (torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1))


def Entrenar(x, y, unidades, capas, dropout, lr):
    torch.manual_seed(SEMILLA)
    modelo = RedLstm(unidades, capas, dropout)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.MSELoss()
    modelo.train()
    for _ in range(EPOCAS):
        optimizador.zero_grad()
        error = criterio(modelo(x), y)
        error.backward()
        optimizador.step()
    return modelo, float(error.detach())


def Mae(real, pred):
    return float(np.mean(np.abs(real - pred)))


def Rmse(real, pred):
    return float(np.sqrt(np.mean((real - pred) ** 2)))


def Combinaciones(rejilla):
    claves = list(rejilla)
    return [dict(zip(claves, v)) for v in itertools.product(*rejilla.values())]


def Tunear(train):
    ajuste = train.iloc[:-VALIDACION]
    escalador = MinMaxScaler().fit(ajuste.values.reshape(-1, 1))
    escalado = escalador.transform(train.values.reshape(-1, 1)).ravel()
    real = train.values[-VALIDACION:]

    filas = []
    for nombre, rejilla in CONFIGURACIONES.items():
        for p in Combinaciones(rejilla):
            x, y = CrearSecuencias(escalado, p['ventana'])
            corte = len(y) - VALIDACION
            modelo, perdida = Entrenar(x[:corte], y[:corte], p['unidades'],
                                       p['capas'], p['dropout'], p['lr'])
            modelo.eval()
            with torch.no_grad():
                pred = modelo(x[corte:]).numpy().reshape(-1, 1)
            pred = escalador.inverse_transform(pred).ravel()
            filas.append({'Configuracion': nombre, **p,
                          'PerdidaEntrenamiento': round(perdida, 6),
                          'ValMae': round(Mae(real, pred), 2),
                          'ValRmse': round(Rmse(real, pred), 2)})
    return pd.DataFrame(filas)


def main():
    os.makedirs('results', exist_ok=True)
    np.random.seed(SEMILLA)

    tablas = []
    for nombre, (train, _) in ObtenerConjuntos().items():
        tabla = Tunear(train)
        tabla.insert(0, 'Serie', nombre)
        tablas.append(tabla)
        print(f'\n{nombre}')
        print(tabla.sort_values('ValRmse').to_string(index=False))

    tuneo = pd.concat(tablas, ignore_index=True)
    tuneo.to_csv('results/TuneoLstm.csv', index=False)

    mejores = tuneo.loc[tuneo.groupby(['Serie', 'Configuracion'])['ValRmse'].idxmin()]
    mejores.to_csv('results/MejoresLstm.csv', index=False)
    print('\nMejor modelo por configuracion')
    print(mejores.to_string(index=False))


if __name__ == '__main__':
    main()
