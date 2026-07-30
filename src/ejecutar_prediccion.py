import os
import sys
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Rutas relativas a la raiz del repositorio
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DATA_DIR    = os.path.join(_ROOT, 'data')
_RESULTS_DIR = os.path.join(_ROOT, 'results')

# Configuración global
SEMILLA = 42
EPOCAS = 500
VALIDACION = 24
TRAIN_RATIO = 0.7
SERIES = ['Total_Consistent', 'Via_Aerea']

# Definición de la grilla de hiperparámetros
CONFIGURACIONES = {
    'LstmSimple': {
        'capas': [1],
        'dropout': [0.0],
        'ventana': [12, 24],
        'unidades': [32, 64],
        'lr': [0.01, 0.001]
    },
    'LstmApilado': {
        'capas': [2],
        'dropout': [0.2],
        'ventana': [12, 24],
        'unidades': [32, 64],
        'lr': [0.01, 0.001]
    },
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

def cargar_series(ruta=None):
    if ruta is None:
        ruta = os.path.join(_DATA_DIR, 'series_de_tiempo_completas.csv')
    return pd.read_csv(ruta, index_col='Fecha', parse_dates=True).asfreq('MS')

def fecha_corte(df):
    fechas = sorted(df.index.unique())
    return pd.Timestamp(fechas[int(len(fechas) * TRAIN_RATIO)])

def dividir_serie(df, columna, corte):
    s = df[columna].interpolate(method='time').fillna(0).clip(lower=0)
    return s.loc[:corte].asfreq('MS'), s.loc[corte:].iloc[1:].asfreq('MS')

def obtener_conjuntos(ruta='data/series_de_tiempo_completas.csv'):
    df = cargar_series(ruta)
    corte = fecha_corte(df)
    return {c: dividir_serie(df, c, corte) for c in SERIES}

def crear_secuencias(valores, ventana):
    x = np.array([valores[i:i + ventana] for i in range(len(valores) - ventana)])
    y = np.array([valores[i + ventana] for i in range(len(valores) - ventana)])
    return (torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1))

def entrenar(x, y, unidades, capas, dropout, lr):
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

def mae(real, pred):
    return float(np.mean(np.abs(real - pred)))

def rmse(real, pred):
    return float(np.sqrt(np.mean((real - pred) ** 2)))

def combinaciones(rejilla):
    claves = list(rejilla)
    return [dict(zip(claves, v)) for v in itertools.product(*rejilla.values())]

def tunear(train):
    ajuste = train.iloc[:-VALIDACION]
    escalador = MinMaxScaler().fit(ajuste.values.reshape(-1, 1))
    escalado = escalador.transform(train.values.reshape(-1, 1)).ravel()
    real = train.values[-VALIDACION:]

    filas = []
    for nombre, rejilla in CONFIGURACIONES.items():
        for p in combinaciones(rejilla):
            x, y = crear_secuencias(escalado, p['ventana'])
            corte = len(y) - VALIDACION
            modelo, perdida = entrenar(x[:corte], y[:corte], p['unidades'],
                                       p['capas'], p['dropout'], p['lr'])
            modelo.eval()
            with torch.no_grad():
                pred = modelo(x[corte:]).numpy().reshape(-1, 1)
            pred = escalador.inverse_transform(pred).ravel()
            filas.append({'Configuracion': nombre, **p,
                          'PerdidaEntrenamiento': round(perdida, 6),
                          'ValMae': round(mae(real, pred), 2),
                          'ValRmse': round(rmse(real, pred), 2)})
    return pd.DataFrame(filas)

def predecir_test(train, test, p, autoregresivo=True):
    torch.manual_seed(SEMILLA)
    escalador = MinMaxScaler().fit(train.values.reshape(-1, 1))
    train_esc = escalador.transform(train.values.reshape(-1, 1)).ravel()
    test_esc = escalador.transform(test.values.reshape(-1, 1)).ravel()
    
    ventana = p['ventana']
    x_tr, y_tr = crear_secuencias(train_esc, ventana)
    
    modelo, _ = entrenar(x_tr, y_tr, p['unidades'], p['capas'], p['dropout'], p['lr'])
    modelo.eval()
    
    if autoregresivo:
        historia = list(train_esc[-ventana:])
        preds_esc = []
        for i in range(len(test)):
            inp = torch.tensor(historia[-ventana:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            with torch.no_grad():
                out = modelo(inp).item()
            preds_esc.append(out)
            historia.append(out)
        preds = escalador.inverse_transform(np.array(preds_esc).reshape(-1, 1)).ravel()
    else:
        full_esc = np.concatenate([train_esc[-ventana:], test_esc])
        x_te = np.array([full_esc[i:i + ventana] for i in range(len(test))])
        inp = torch.tensor(x_te, dtype=torch.float32).unsqueeze(-1)
        with torch.no_grad():
            preds_esc = modelo(inp).numpy().reshape(-1, 1)
        preds = escalador.inverse_transform(preds_esc).ravel()
        
    return preds

def main():
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    np.random.seed(SEMILLA)
    
    conjuntos = obtener_conjuntos()
    
    print("=== 1. TUNEO DE PARÁMETROS ===")
    tablas_tuneo = []
    for nombre, (train, _) in conjuntos.items():
        tabla = tunear(train)
        tabla.insert(0, 'Serie', nombre)
        tablas_tuneo.append(tabla)
        print(f'\nResultados de Tuneo para {nombre}:')
        print(tabla.sort_values('ValRmse').to_string(index=False))

    tuneo_df = pd.concat(tablas_tuneo, ignore_index=True)
    tuneo_df.to_csv(os.path.join(_RESULTS_DIR, 'TuneoLstm.csv'), index=False)

    # Seleccionar el mejor hiperparámetro por serie
    mejores_modelos = tuneo_df.loc[tuneo_df.groupby('Serie')['ValRmse'].idxmin()]
    mejores_modelos.to_csv(os.path.join(_RESULTS_DIR, 'MejoresLstm.csv'), index=False)
    print('\n=== MEJORES MODELOS SELECCIONADOS POR VALIDACIÓN ===')
    print(mejores_modelos.to_string(index=False))
    
    print('\n=== 2. PREDICCIÓN Y EVALUACIÓN EN CONJUNTO DE PRUEBA (TEST) ===')
    resultados_test = []
    
    plt.figure(figsize=(14, 10))
    
    for idx, row in mejores_modelos.iterrows():
        serie = row['Serie']
        train, test = conjuntos[serie]
        params = {
            'unidades': int(row['unidades']),
            'capas': int(row['capas']),
            'dropout': float(row['dropout']),
            'ventana': int(row['ventana']),
            'lr': float(row['lr'])
        }
        
        # Predicción Autoregresiva (Multi-step)
        preds_ar = predecir_test(train, test, params, autoregresivo=True)
        mae_ar = mae(test.values, preds_ar)
        rmse_ar = rmse(test.values, preds_ar)
        
        # Predicción One-step ahead (Rolling)
        preds_os = predecir_test(train, test, params, autoregresivo=False)
        mae_os = mae(test.values, preds_os)
        rmse_os = rmse(test.values, preds_os)
        
        resultados_test.append({
            'Serie': serie,
            'Configuracion': row['Configuracion'],
            'Ventana': params['ventana'],
            'Unidades': params['unidades'],
            'LR': params['lr'],
            'Test_MAE_Autoregresivo': round(mae_ar, 2),
            'Test_RMSE_Autoregresivo': round(rmse_ar, 2),
            'Test_MAE_OneStep': round(mae_os, 2),
            'Test_RMSE_OneStep': round(rmse_os, 2)
        })
        
        # Graficar resultados
        plt.subplot(2, 1, 1 if serie == 'Total_Consistent' else 2)
        plt.plot(train.index[-36:], train.values[-36:], label='Entrenamiento (Últimos 3 años)', color='black', alpha=0.6)
        plt.plot(test.index, test.values, label='Test Real', color='blue', linewidth=2)
        plt.plot(test.index, preds_ar, label='LSTM Autoregresivo (Multi-step)', color='red', linestyle='--')
        plt.plot(test.index, preds_os, label='LSTM One-Step Ahead', color='green', linestyle=':')
        plt.title(f'Predicciones del Mejor Modelo LSTM - Serie: {serie}')
        plt.xlabel('Fecha')
        plt.ylabel('Visitantes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Guardar predicciones en CSV
        df_preds = pd.DataFrame({
            'Fecha': test.index,
            'Real': test.values,
            'Pred_Autoregresivo': preds_ar,
            'Pred_OneStep': preds_os
        })
        df_preds.to_csv(os.path.join(_RESULTS_DIR, f'predicciones_{serie.lower()}.csv'), index=False)
        print(f'  -> Guardado results/predicciones_{serie.lower()}.csv ({len(df_preds)} filas)')
        
    plt.tight_layout()
    plt.savefig(os.path.join(_RESULTS_DIR, 'predicciones_lstm.png'), dpi=300)
    plt.close()
    
    df_res_test = pd.DataFrame(resultados_test)
    df_res_test.to_csv(os.path.join(_RESULTS_DIR, 'ResultadosTestLSTM.csv'), index=False)
    print('\nResultados Finales en Test:')
    print(df_res_test.to_string(index=False))

if __name__ == '__main__':
    main()
