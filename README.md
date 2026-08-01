# Laboratorio 2 — Deep Learning y Análisis de Series de Tiempo

**CC3084 – Data Science · Universidad del Valle de Guatemala · Semestre II 2026**

## Integrantes

- Javier España #23361
- Ángel Esquit #23221
- Roberto Barreda #23354

## Link al Repositorio

- GitHub: [https://github.com/Javier-Espana/Lab2-DS](https://github.com/Javier-Espana/Lab2-DS)

---

## Estructura del Repositorio

```
Lab2-DS/
├── data/                                 # Conjuntos de datos de entrada
│   ├── series_de_tiempo_completas.csv    # Dataset principal (7 series, 210 meses)
│   ├── lab1_model_comparison_metrics.csv # Métricas de referencia de Lab 1
│   ├── train_total_consistent.csv
│   ├── test_total_consistent.csv
│   ├── train_via_aerea.csv
│   └── test_via_aerea.csv
│
├── docs/                                 # Documentación y PDF de la guía
│   ├── Laboratorio 2. Deep Learning_Series. 2026.pdf
│   └── Informe_Lab02.tex                # Documento LaTeX del informe final
│
├── notebooks/                            # Jupyter Notebooks organizados por ejercicio
│   ├── 01_Ejercicio1_LSTM.ipynb          # Ejercicio 1: Modelos LSTM (1.1 → 1.4)
│   └── 02_Ejercicio2_Catch22.ipynb       # Ejercicio 2: Análisis catch22 + LSTM Híbrido (2.1 → 2.14)
│
├── src/                                  # Scripts y módulos de Python
│   ├── preparacion_datos.py              # División 70/30 train/test
│   ├── modelos_lstm.py                   # Definición de red LSTM + tuneo
│   ├── ejecutar_prediccion.py            # Pipeline de predicción Ejercicio 1
│   ├── ExtraccionCatch22.py              # Extracción de las 22 características canónicas
│   ├── AnalisisCatch22.py                # PCA, Clustering, Heatmap, Distancias, Correlaciones
│   └── lstm_catch22.py                   # Modelo híbrido Catch22-LSTM (Ejercicio 2.14)
│
├── results/                              # Artefactos, tablas de métricas y gráficos
│   ├── TuneoLstm.csv
│   ├── MejoresLstm.csv
│   ├── ResultadosTestLSTM.csv
│   ├── Catch22Features.csv
│   ├── Catch22Estandarizado.csv
│   ├── Catch22Grupos.csv
│   ├── Catch22PcaVarianza.csv
│   ├── Catch22PcaCargas.csv
│   ├── Catch22Correlaciones.csv
│   ├── Catch22Distancias.csv
│   ├── ResultadosCatch22LSTM.csv
│   ├── Comparacion_LSTM_vs_Catch22LSTM.csv
│   ├── predicciones_lstm.png
│   ├── Catch22Pca.png
│   ├── Catch22Dendrograma.png
│   ├── Catch22Heatmap.png
│   ├── Catch22Correlaciones.png
│   ├── Catch22Distancias.png
│   └── predicciones_catch22_lstm.png
│
├── requirements.txt
└── README.md
```

---

## Contenido de los Ejercicios

### Ejercicio 1: Modelado con LSTM
- **1.1 Preparación:** División 70/30 train/test estricta.
- **1.2 Tuneo:** Grid search de 16 configuraciones (LstmSimple vs. LstmApilado) por serie sobre 24 meses de validación interna.
- **1.3 Predicción:** Evaluación en test con estrategias Multi-step (autoregresivo) y One-step ahead.
- **1.4 Comparación:** Evaluación contra modelos clásicos de Lab 1 (Holt-Winters, SARIMA, Prophet).

### Ejercicio 2: Similitud con catch22 y LSTM Híbrido
- **2.1 a 2.5:** Fundamentos de catch22, extracción de 22 características para 7 series, estandarización $z$-score, PCA (69.5% varianza explicada en PC1-PC2), Clustering Jerárquico de Ward ($k=2$), Heatmap, Correlaciones de Pearson y Matriz de Distancias.
- **2.6 a 2.13:** Respuestas detalladas sobre similitud dinámica, interpretación de cargas en PCA, identificación del outlier estructural (`Pais_Guatemala`), consistencia con EDA tradicional y nuevos descubrimientos.
- **2.14:** Red híbrida **Catch22-LSTM** combinando la secuencia temporal con el prior global de catch22.

---

## Instrucciones de Ejecución

### 1. Instalación de dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar Notebooks en Jupyter
```bash
jupyter notebook notebooks/01_Ejercicio1_LSTM.ipynb
jupyter notebook notebooks/02_Ejercicio2_Catch22.ipynb
```

### 3. Ejecutar Scripts de Python Standalone
```bash
# Ejercicio 1: Predicción LSTM baseline
python src/ejecutar_prediccion.py

# Ejercicio 2: Análisis exploratorio de catch22
python src/AnalisisCatch22.py

# Ejercicio 2.14: Modelo híbrido Catch22-LSTM
python src/lstm_catch22.py
```
