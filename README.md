# Laboratorio 2 — Deep Learning: Series de Tiempo

**CC3084 – Data Science · Universidad del Valle de Guatemala · Semestre II 2026**

## Integrantes

- Javier España #23361
- Ángel Esquit #23221
- Roberto Barreda #23354

## Link al REPO

Link: https://github.com/Javier-Espana/Lab2-DS

---

## Estructura del Repositorio

```
Lab2-DS/
├── data/                              # Datos de entrada
│   ├── series_de_tiempo_completas.csv # CSV principal con todas las series
│   ├── lab1_model_comparison_metrics.csv # Métricas Lab 1 para comparación
│   ├── train_total_consistent.csv
│   ├── test_total_consistent.csv
│   ├── train_via_aerea.csv
│   └── test_via_aerea.csv
│
├── notebooks/                         # Jupyter Notebooks (ordenados)
│   └── 01_Ejercicio1_LSTM.ipynb       # Ejercicio 1 completo (1.1 → 1.4)
│
├── src/                               # Módulos Python reutilizables
│   ├── preparacion_datos.py           # Carga y división train/test
│   ├── modelos_lstm.py                # Arquitectura LSTM + tuneo
│   └── ejecutar_prediccion.py         # Pipeline completo de predicción
│
├── results/                           # Salidas generadas por los notebooks/scripts
│   ├── TuneoLstm.csv
│   ├── MejoresLstm.csv
│   ├── ResultadosTestLSTM.csv
│   ├── comparacion_lstm_vs_lab1.csv
│   ├── predicciones_total_consistent.csv
│   ├── predicciones_via_aerea.csv
│   ├── predicciones_lstm.png
│   ├── tuneo_barras.png
│   ├── comparacion_rmse_lab1_vs_lstm.png
│   └── error_absoluto_lstm.png
│
├── requirements.txt
└── README.md
```

---

## Ejercicio 1: Modelos LSTM

El notebook `notebooks/01_Ejercicio1_LSTM.ipynb` cubre:

| Sección | Contenido |
|---------|-----------|
| **1.1** | Carga del CSV del Lab 1, división 70/30 train/test |
| **1.2** | Dos arquitecturas LSTM (Simple y Apilada) + tuneo de hiperparámetros (16 combinaciones × 2 series) |
| **1.3** | Predicción con el mejor modelo: Multi-step (autoregresivo) y One-step ahead |
| **1.4** | Análisis comparativo: ¿cuál serie predijo mejor? ¿LSTM supera a Lab 1? |

## Cómo ejecutar

### Desde Jupyter (recomendado)
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_Ejercicio1_LSTM.ipynb
```

### Script standalone
```bash
pip install -r requirements.txt
python src/ejecutar_prediccion.py
```
