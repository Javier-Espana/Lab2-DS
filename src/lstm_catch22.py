import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Paths relative to repository root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DATA_DIR = os.path.join(_ROOT, 'data')
_RESULTS_DIR = os.path.join(_ROOT, 'results')

SEMILLA = 42
EPOCAS = 500
TRAIN_RATIO = 0.70
SERIES = ['Total_Consistent', 'Via_Aerea']


class Catch22LSTM(nn.Module):
    """
    Red LSTM híbrida que procesa la secuencia temporal y la combina con
    las 22 características canónicas de catch22.
    """
    def __init__(self, num_catch22=22, unidades=32, capas=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(1, unidades, capas, batch_first=True,
                            dropout=dropout if capas > 1 else 0.0)
        self.fc_combinado = nn.Sequential(
            nn.Linear(unidades + num_catch22, unidades),
            nn.ReLU(),
            nn.Linear(unidades, 1)
        )

    def forward(self, x_seq, x_c22):
        # x_seq: (batch, seq_len, 1)
        # x_c22: (batch, num_catch22)
        lstm_out, _ = self.lstm(x_seq)
        h_last = lstm_out[:, -1, :]  # (batch, unidades)
        combined = torch.cat([h_last, x_c22], dim=1)
        return self.fc_combinado(combined)


def CargarSeries():
    ruta = os.path.join(_DATA_DIR, 'series_de_tiempo_completas.csv')
    return pd.read_csv(ruta, index_col='Fecha', parse_dates=True).asfreq('MS')


def CargarCatch22Estandarizado():
    ruta = os.path.join(_RESULTS_DIR, 'Catch22Estandarizado.csv')
    if os.path.exists(ruta):
        return pd.read_csv(ruta, index_col='Serie')
    else:
        # Si no existe, cargar Catch22Features y estandarizar
        ruta_raw = os.path.join(_RESULTS_DIR, 'Catch22Features.csv')
        df_raw = pd.read_csv(ruta_raw, index_col='Serie')
        scaled = StandardScaler().fit_transform(df_raw.values)
        return pd.DataFrame(scaled, index=df_raw.index, columns=df_raw.columns)


def DividirSerie(df, columna):
    fechas = sorted(df.index.unique())
    corte = pd.Timestamp(fechas[int(len(fechas) * TRAIN_RATIO)])
    s = df[columna].interpolate(method='time').fillna(0).clip(lower=0)
    return s.loc[:corte].asfreq('MS'), s.loc[corte:].iloc[1:].asfreq('MS')


def CrearSecuencias(valores, ventana):
    x = np.array([valores[i:i + ventana] for i in range(len(valores) - ventana)])
    y = np.array([valores[i + ventana] for i in range(len(valores) - ventana)])
    return (torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1))


def EntrenarCatch22LSTM(x_tr, y_tr, c22_vector, unidades, capas, dropout, lr):
    torch.manual_seed(SEMILLA)
    num_c22 = len(c22_vector)
    modelo = Catch22LSTM(num_catch22=num_c22, unidades=unidades, capas=capas, dropout=dropout)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    crit = nn.MSELoss()

    c22_tensor_tr = torch.tensor(
        np.tile(c22_vector, (len(x_tr), 1)), dtype=torch.float32
    )

    modelo.train()
    for _ in range(EPOCAS):
        opt.zero_grad()
        preds = modelo(x_tr, c22_tensor_tr)
        loss = crit(preds, y_tr)
        loss.backward()
        opt.step()
    return modelo, float(loss.detach())


def CalcMae(real, pred):
    return float(np.mean(np.abs(real - pred)))


def CalcRmse(real, pred):
    return float(np.sqrt(np.mean((real - pred) ** 2)))


def PredecirTest(train, test, c22_vec, params, autoregresivo=True):
    torch.manual_seed(SEMILLA)
    escalador = MinMaxScaler().fit(train.values.reshape(-1, 1))
    train_esc = escalador.transform(train.values.reshape(-1, 1)).ravel()
    test_esc = escalador.transform(test.values.reshape(-1, 1)).ravel()

    ventana = params['ventana']
    x_tr, y_tr = CrearSecuencias(train_esc, ventana)

    modelo, _ = EntrenarCatch22LSTM(
        x_tr, y_tr, c22_vec,
        params['unidades'], params['capas'], params['dropout'], params['lr']
    )
    modelo.eval()

    c22_tensor_single = torch.tensor(c22_vec.reshape(1, -1), dtype=torch.float32)

    if autoregresivo:
        historia = list(train_esc[-ventana:])
        preds_esc = []
        for _ in range(len(test)):
            inp = torch.tensor(historia[-ventana:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            with torch.no_grad():
                out = modelo(inp, c22_tensor_single).item()
            preds_esc.append(out)
            historia.append(out)
        preds = escalador.inverse_transform(np.array(preds_esc).reshape(-1, 1)).ravel()
    else:
        full_esc = np.concatenate([train_esc[-ventana:], test_esc])
        x_te = np.array([full_esc[i:i + ventana] for i in range(len(test))])
        inp = torch.tensor(x_te, dtype=torch.float32).unsqueeze(-1)
        c22_tensor_te = torch.tensor(np.tile(c22_vec, (len(test), 1)), dtype=torch.float32)
        with torch.no_grad():
            preds_esc = modelo(inp, c22_tensor_te).numpy().reshape(-1, 1)
        preds = escalador.inverse_transform(preds_esc).ravel()

    return preds


def main():
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    np.random.seed(SEMILLA)

    df_series = CargarSeries()
    df_c22 = CargarCatch22Estandarizado()
    df_mejores_ej1 = pd.read_csv(os.path.join(_RESULTS_DIR, 'MejoresLstm.csv'))

    resultados = []
    plt.figure(figsize=(14, 10))

    for idx, serie in enumerate(SERIES):
        train, test = DividirSerie(df_series, serie)
        c22_vec = df_c22.loc[serie].values

        # Usar la mejor parametrización encontrada en el Ejercicio 1 para esa serie
        best_row = df_mejores_ej1[df_mejores_ej1['Serie'] == serie].iloc[0]
        params = {
            'ventana': int(best_row['ventana']),
            'unidades': int(best_row['unidades']),
            'capas': int(best_row['capas']),
            'dropout': float(best_row['dropout']),
            'lr': float(best_row['lr']),
        }

        print(f"\n--- Entrenando Catch22-LSTM para {serie} ---")
        print(f"Parámetros: {params}")

        preds_ar = PredecirTest(train, test, c22_vec, params, autoregresivo=True)
        preds_os = PredecirTest(train, test, c22_vec, params, autoregresivo=False)

        mae_ar = CalcMae(test.values, preds_ar)
        rmse_ar = CalcRmse(test.values, preds_ar)
        mae_os = CalcMae(test.values, preds_os)
        rmse_os = CalcRmse(test.values, preds_os)

        resultados.append({
            'Serie': serie,
            'Modelo': 'Catch22-LSTM',
            'Ventana': params['ventana'],
            'Unidades': params['unidades'],
            'LR': params['lr'],
            'Test_MAE_Multistep': round(mae_ar, 2),
            'Test_RMSE_Multistep': round(rmse_ar, 2),
            'Test_MAE_OneStep': round(mae_os, 2),
            'Test_RMSE_OneStep': round(rmse_os, 2),
        })

        # Guardar CSV de predicciones
        df_preds = pd.DataFrame({
            'Fecha': test.index,
            'Real': test.values,
            'Pred_Catch22_Multistep': preds_ar,
            'Pred_Catch22_OneStep': preds_os,
        })
        df_preds.to_csv(os.path.join(_RESULTS_DIR, f'predicciones_catch22_lstm_{serie.lower()}.csv'), index=False)

        # Plot
        plt.subplot(2, 1, idx + 1)
        plt.plot(train.index[-36:], train.values[-36:], label='Train (Últimos 3 años)', color='#64748b', alpha=0.7)
        plt.plot(test.index, test.values, label='Test Real', color='#2563eb', linewidth=2)
        plt.plot(test.index, preds_ar, label='Catch22-LSTM Multi-step', color='#dc2626', linestyle='--')
        plt.plot(test.index, preds_os, label='Catch22-LSTM One-step', color='#16a34a', linestyle=':')
        plt.title(f'Predicciones Modelo Catch22-LSTM - Serie: {serie}', fontsize=12, fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Visitantes')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(_RESULTS_DIR, 'predicciones_catch22_lstm.png'), dpi=150)
    plt.close()

    df_res = pd.DataFrame(resultados)
    df_res.to_csv(os.path.join(_RESULTS_DIR, 'ResultadosCatch22LSTM.csv'), index=False)

    print("\n=== RESULTADOS CATCH22-LSTM EN TEST ===")
    print(df_res.to_string(index=False))

    # Cargar y comparar con Ejercicio 1
    df_ej1_res = pd.read_csv(os.path.join(_RESULTS_DIR, 'ResultadosTestLSTM.csv'))

    comparativa = []
    for serie in SERIES:
        ej1_row = df_ej1_res[df_ej1_res['Serie'] == serie].iloc[0]
        c22_row = df_res[df_res['Serie'] == serie].iloc[0]

        rmse_ej1_ms = ej1_row.get('Test_RMSE_Multistep', ej1_row.get('Test_RMSE_Autoregresivo'))
        rmse_c22_ms = c22_row['Test_RMSE_Multistep']
        rmse_ej1_os = ej1_row['Test_RMSE_OneStep']
        rmse_c22_os = c22_row['Test_RMSE_OneStep']

        mejora_ms = round((rmse_ej1_ms - rmse_c22_ms) / rmse_ej1_ms * 100, 2)
        mejora_os = round((rmse_ej1_os - rmse_c22_os) / rmse_ej1_os * 100, 2)

        comparativa.append({
            'Serie': serie,
            'LSTM_Ej1_RMSE_MS': rmse_ej1_ms,
            'Catch22_LSTM_RMSE_MS': rmse_c22_ms,
            'Mejora_MS_%': mejora_ms,
            'LSTM_Ej1_RMSE_OS': rmse_ej1_os,
            'Catch22_LSTM_RMSE_OS': rmse_c22_os,
            'Mejora_OS_%': mejora_os,
        })

    df_comp = pd.DataFrame(comparativa)
    df_comp.to_csv(os.path.join(_RESULTS_DIR, 'Comparacion_LSTM_vs_Catch22LSTM.csv'), index=False)
    print("\n=== COMPARACIÓN BASELINE LSTM (EJ 1) VS CATCH22-LSTM (EJ 2.14) ===")
    print(df_comp.to_string(index=False))


if __name__ == '__main__':
    main()
