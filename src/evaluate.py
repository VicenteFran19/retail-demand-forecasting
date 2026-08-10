"""
Metricas de evaluacion para forecasting de demanda.

WMAPE (Weighted Mean Absolute Percentage Error) se prefiere sobre MAPE
simple en retail porque no se rompe cuando hay dias con demanda = 0
(muy comun por producto/tienda/dia), y pondera los errores segun el
volumen real de cada serie, no cada punto por igual.
"""

import numpy as np
import pandas as pd


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominador = np.sum(np.abs(y_true))
    if denominador == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denominador * 100)


def resumen_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)  # no tiene sentido predecir ventas negativas

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    w = wmape(y_true, y_pred)

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "WMAPE_%": round(w, 4)}
