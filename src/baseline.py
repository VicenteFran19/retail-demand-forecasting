"""
Baseline estacional: predice la demanda del dia D como el promedio de ese
mismo dia de la semana en las ultimas 4 ocurrencias (naive estacional).

Es el estandar minimo que cualquier modelo de ML debe superar para
justificar su complejidad. Si LightGBM no le gana a esto por un margen
claro, no vale la pena el modelo.
"""

import pandas as pd
import numpy as np


def prediccion_baseline_estacional(df: pd.DataFrame, columna_objetivo: str = "unidades_vendidas") -> pd.Series:
    """
    Para cada fila (tienda, producto, fecha), predice el promedio de las
    ultimas 4 observaciones del MISMO dia de la semana, usando solo datos
    pasados (shift adecuado, sin fuga de informacion).
    """
    df = df.sort_values(["tienda_id", "producto_id", "fecha"]).copy()
    df["dow"] = df["fecha"].dt.dayofweek

    pred = (
        df.groupby(["tienda_id", "producto_id", "dow"])[columna_objetivo]
        .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    )
    return pred.fillna(df.groupby(["tienda_id", "producto_id"])[columna_objetivo].transform("mean"))
