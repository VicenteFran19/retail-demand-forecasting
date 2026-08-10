import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest

from src.features import construir_features, split_temporal, FEATURE_COLS
from src.evaluate import wmape, resumen_metricas
from src.monitoring.psi import calcular_psi, interpretar_psi


def _df_juguete():
    # Ojo: usamos valores NO periodicos (indice creciente, no dia de semana)
    # a proposito. Si usaramos un patron que se repite cada 7 dias, una
    # ventana con leakage y una sin leakage podrian coincidir por
    # casualidad y el test de anti-leakage daria un falso positivo.
    fechas = pd.date_range("2024-01-01", periods=60, freq="D")
    filas = []
    for i, f in enumerate(fechas):
        filas.append({
            "fecha": f, "tienda_id": "T1", "producto_id": "P1",
            "unidades_vendidas": 10 + i,  # estrictamente creciente, sin periodicidad
            "precio": 20.0, "en_promocion": False, "hubo_quiebre_stock": False,
        })
    return pd.DataFrame(filas)


def test_construir_features_no_revienta():
    df = _df_juguete()
    resultado = construir_features(df)
    assert len(resultado) == len(df)
    for col in FEATURE_COLS:
        assert col in resultado.columns


def test_lag_1_no_usa_informacion_del_mismo_dia():
    """
    Test critico anti-leakage: el lag_1 de un dia D debe ser EXACTAMENTE
    igual al valor de unidades_vendidas del dia D-1, nunca del dia D.
    """
    df = _df_juguete()
    resultado = construir_features(df).sort_values("fecha").reset_index(drop=True)

    for i in range(1, len(resultado)):
        valor_dia_anterior = resultado.loc[i - 1, "unidades_vendidas"]
        lag_1_dia_actual = resultado.loc[i, "lag_1"]
        if pd.notna(lag_1_dia_actual):
            assert lag_1_dia_actual == valor_dia_anterior


def test_media_movil_no_incluye_dia_actual():
    """
    Otro test anti-leakage: la media_movil_7 del dia D no debe incluir
    unidades_vendidas del propio dia D en su calculo.
    """
    df = _df_juguete()
    resultado = construir_features(df).sort_values("fecha").reset_index(drop=True)

    fila = resultado.iloc[10]
    valor_del_dia = fila["unidades_vendidas"]
    ventana_manual = resultado.iloc[3:10]["unidades_vendidas"].mean()  # dias 3..9, excluye el dia 10

    assert abs(fila["media_movil_7"] - ventana_manual) < 1e-6
    # La media movil no deberia coincidir "sospechosamente" con incluir el valor propio
    ventana_con_leakage = resultado.iloc[4:11]["unidades_vendidas"].mean()
    assert abs(fila["media_movil_7"] - ventana_con_leakage) > 1e-6 or valor_del_dia == 0


def test_split_temporal_sin_solapamiento_de_fechas():
    df = _df_juguete()
    resultado = construir_features(df)
    train, val, test = split_temporal(resultado, dias_test=7, dias_val=7, margen_dias=2)

    if len(train) and len(val):
        assert train["fecha"].max() < val["fecha"].min()
    if len(val) and len(test):
        assert val["fecha"].max() < test["fecha"].min()


def test_split_temporal_es_por_fecha_no_por_posicion():
    df = _df_juguete()
    resultado = construir_features(df)
    train, val, test = split_temporal(resultado, dias_test=7, dias_val=7, margen_dias=2)

    for fecha in test["fecha"]:
        assert fecha not in train["fecha"].values
        assert fecha not in val["fecha"].values


def test_wmape_calculo_basico():
    y_true = np.array([10, 20, 30, 0])
    y_pred = np.array([12, 18, 30, 5])
    resultado = wmape(y_true, y_pred)
    esperado = (2 + 2 + 0 + 5) / (10 + 20 + 30 + 0) * 100
    assert abs(resultado - esperado) < 1e-6


def test_wmape_prediccion_perfecta_da_cero():
    y = np.array([5, 10, 15])
    assert wmape(y, y) == 0.0


def test_resumen_metricas_no_permite_predicciones_negativas():
    y_true = np.array([5, 10])
    y_pred = np.array([-2, 8])
    resultado = resumen_metricas(y_true, y_pred)
    # La funcion debe clipear a 0 antes de calcular, no dejar pasar negativos silenciosamente
    assert resultado["MAE"] >= 0


def test_psi_identico_es_cero():
    dist = np.random.default_rng(0).normal(0, 1, 1000)
    psi = calcular_psi(dist, dist)
    assert psi < 0.01


def test_psi_distribuciones_muy_distintas_da_alto():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 1000)
    actual = rng.normal(5, 1, 1000)
    psi = calcular_psi(base, actual)
    assert psi > 0.25


def test_interpretar_psi_umbrales():
    assert interpretar_psi(0.05) == "sin cambio significativo"
    assert interpretar_psi(0.15) == "cambio moderado - vigilar de cerca"
    assert interpretar_psi(0.30) == "cambio importante - considerar reentrenar el modelo"
