# Laboratorio 2 - Deep Learning: Series de Tiempo

**CC3084 - Data Science · Universidad del Valle de Guatemala · Semestre II 2026**

## Integrantes

- Javier España #23361
- Ángel Esquit #23221
- Roberto Barreda #23354

## Link al REPO

Link: https://github.com/Javier-Espana/Lab2-DS

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
