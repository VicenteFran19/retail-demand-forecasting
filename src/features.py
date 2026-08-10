"""
Feature engineering para forecasting de demanda.

Regla de oro contra fuga de informacion (data leakage):
Para predecir el dia D, SOLO podemos usar informacion disponible ANTES de D.
Por eso todo lag y ventana movil usa .shift(1) como minimo antes de calcular
cualquier ventana: la ventana movil de 7 dias para el dia D describe los
7 dias ANTERIORES a D-1, nunca incluye D.
"""

import pandas as pd
import numpy as np


LAGS = [1, 7, 14, 28]
VENTANAS = [7, 14, 28]


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["tienda_id", "producto_id", "fecha"]).reset_index(drop=True)

    grupo = df.groupby(["tienda_id", "producto_id"])["unidades_vendidas"]

    # --- Lags: valores pasados directos ---
    for lag in LAGS:
        df[f"lag_{lag}"] = grupo.shift(lag)

    # --- Ventanas moviles: SIEMPRE sobre la serie ya desplazada 1 dia ---
    # shift(1) primero garantiza que la ventana del dia D no incluye el dia D
    df["_serie_shift1"] = grupo.shift(1)
    grupo_shift = df.groupby(["tienda_id", "producto_id"])["_serie_shift1"]
    for ventana in VENTANAS:
        df[f"media_movil_{ventana}"] = grupo_shift.transform(
            lambda s: s.rolling(ventana).mean()
        )
        df[f"std_movil_{ventana}"] = grupo_shift.transform(
            lambda s: s.rolling(ventana).std()
        )
    df = df.drop(columns=["_serie_shift1"])

    # --- Variables de calendario (estas SI se conocen de antemano, no hay leakage) ---
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["mes"] = df["fecha"].dt.month
    df["es_fin_de_semana"] = (df["dia_semana"] >= 4).astype(int)
    df["dia_del_mes"] = df["fecha"].dt.day
    df["semana_del_anio"] = df["fecha"].dt.isocalendar().week.astype(int)

    # --- Precio y promocion: se conocen el dia D (decision de negocio, no leakage) ---
    df["en_promocion"] = df["en_promocion"].astype(int)

    # --- Identificadores categoricos como codigos enteros para LightGBM ---
    df["tienda_cod"] = df["tienda_id"].astype("category").cat.codes
    df["producto_cod"] = df["producto_id"].astype("category").cat.codes

    return df


FEATURE_COLS = (
    [f"lag_{l}" for l in LAGS]
    + [f"media_movil_{v}" for v in VENTANAS]
    + [f"std_movil_{v}" for v in VENTANAS]
    + [
        "dia_semana", "mes", "es_fin_de_semana", "dia_del_mes", "semana_del_anio",
        "precio", "en_promocion", "tienda_cod", "producto_cod",
    ]
)
TARGET_COL = "unidades_vendidas"


def split_temporal(df: pd.DataFrame, dias_test: int = 28, dias_val: int = 28, margen_dias: int = 3):
    """
    Split temporal por FECHA COMPLETA (no por posicion de fila), con margen
    entre conjuntos para evitar que ventanas moviles de un conjunto se
    calculen usando datos justo al borde del otro conjunto.
    """
    fecha_max = df["fecha"].max()
    inicio_test = fecha_max - pd.Timedelta(days=dias_test - 1)
    fin_val = inicio_test - pd.Timedelta(days=margen_dias + 1)
    inicio_val = fin_val - pd.Timedelta(days=dias_val - 1)
    fin_train = inicio_val - pd.Timedelta(days=margen_dias + 1)

    train = df[df["fecha"] <= fin_train].copy()
    val = df[(df["fecha"] >= inicio_val) & (df["fecha"] <= fin_val)].copy()
    test = df[df["fecha"] >= inicio_test].copy()

    return train, val, test
