"""
PSI (Population Stability Index): mide si la distribucion de una variable
cambio entre dos periodos (por ejemplo, entre los datos de entrenamiento
y los datos de produccion de las ultimas semanas).

Regla practica de la industria:
- PSI < 0.10  -> sin cambio significativo
- 0.10 <= PSI < 0.25 -> cambio moderado, vigilar
- PSI >= 0.25 -> cambio importante, considerar reentrenar el modelo
"""

import numpy as np
import pandas as pd


def calcular_psi(distribucion_base: np.ndarray, distribucion_actual: np.ndarray, n_bins: int = 10) -> float:
    distribucion_base = np.asarray(distribucion_base, dtype=float)
    distribucion_actual = np.asarray(distribucion_actual, dtype=float)

    # Bins definidos sobre la distribucion base (la de referencia/entrenamiento)
    _, bordes = np.histogram(distribucion_base, bins=n_bins)
    bordes[0] = -np.inf
    bordes[-1] = np.inf

    frec_base, _ = np.histogram(distribucion_base, bins=bordes)
    frec_actual, _ = np.histogram(distribucion_actual, bins=bordes)

    prop_base = frec_base / frec_base.sum()
    prop_actual = frec_actual / frec_actual.sum()

    # Evitar log(0) / division por cero con un piso minimo
    epsilon = 1e-4
    prop_base = np.clip(prop_base, epsilon, None)
    prop_actual = np.clip(prop_actual, epsilon, None)

    psi_por_bin = (prop_actual - prop_base) * np.log(prop_actual / prop_base)
    return float(np.sum(psi_por_bin))


def interpretar_psi(valor_psi: float) -> str:
    if valor_psi < 0.10:
        return "sin cambio significativo"
    elif valor_psi < 0.25:
        return "cambio moderado - vigilar de cerca"
    else:
        return "cambio importante - considerar reentrenar el modelo"


def reporte_drift(df_base: pd.DataFrame, df_actual: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    filas = []
    for col in columnas:
        psi = calcular_psi(df_base[col].dropna().values, df_actual[col].dropna().values)
        filas.append({"feature": col, "psi": round(psi, 4), "interpretacion": interpretar_psi(psi)})
    return pd.DataFrame(filas).sort_values("psi", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.features import construir_features, split_temporal, FEATURE_COLS

    df = pd.read_parquet("data/ventas_sinteticas.parquet")
    df_feat = construir_features(df)
    train, val, test = split_temporal(df_feat)

    reporte = reporte_drift(train, test, ["precio", "lag_7", "media_movil_28", "en_promocion"])
    print(reporte.to_string(index=False))
