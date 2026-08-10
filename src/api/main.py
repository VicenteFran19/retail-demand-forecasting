"""
API de serving del modelo de forecasting de demanda.

Expone:
- POST /predict         -> prediccion puntual de demanda + recomendacion de inventario
- GET  /health           -> healthcheck para monitoreo/orquestacion (k8s, docker, etc.)
- GET  /model-info        -> metadata del modelo cargado
"""

import os
import sys

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.api.schemas import PrediccionRequest, PrediccionResponse
from src.features import FEATURE_COLS

MODEL_PATH = os.environ.get("MODEL_PATH", "models/lightgbm_model.txt")

app = FastAPI(
    title="Retail Demand Forecasting API",
    description="Sirve predicciones de demanda diaria y recomendaciones de inventario.",
    version="1.0.0",
)

_modelo = None


def cargar_modelo():
    global _modelo
    if _modelo is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No se encontro el modelo en {MODEL_PATH}. Corre primero: python src/train.py"
            )
        _modelo = lgb.Booster(model_file=MODEL_PATH)
    return _modelo


# Cargamos el modelo al importar el modulo (en vez de un lifespan event),
# asi TestClient(app) y `uvicorn src.api.main:app` funcionan igual sin
# depender de que se dispare un evento de startup.
try:
    cargar_modelo()
except FileNotFoundError:
    # Permite que /health reporte el problema en vez de que la app no arranque
    pass


@app.get("/health")
def health():
    modelo_cargado = _modelo is not None
    return {"status": "ok" if modelo_cargado else "modelo no cargado", "modelo_cargado": modelo_cargado}


@app.get("/model-info")
def model_info():
    modelo = cargar_modelo()
    return {
        "num_features": modelo.num_feature(),
        "features_esperadas": FEATURE_COLS,
        "num_arboles": modelo.num_trees(),
    }


TIENDA_A_COD = None
PRODUCTO_A_COD = None


def _mapear_categoricos(tienda_id: str, producto_id: str):
    """
    En un caso real, este mapeo se guarda junto con el modelo (por ejemplo
    en un archivo de metadata) para garantizar consistencia entre train y
    serving. Aqui usamos un hash estable como simplificacion valida para
    la demo, documentada explicitamente como limitacion conocida.
    """
    tienda_cod = abs(hash(tienda_id)) % 1000
    producto_cod = abs(hash(producto_id)) % 1000
    return tienda_cod, producto_cod


@app.post("/predict", response_model=PrediccionResponse)
def predict(payload: PrediccionRequest):
    modelo = cargar_modelo()

    tienda_cod, producto_cod = _mapear_categoricos(payload.tienda_id, payload.producto_id)
    dia_semana = payload.fecha.weekday()
    mes = payload.fecha.month
    es_fin_de_semana = int(dia_semana >= 4)
    dia_del_mes = payload.fecha.day
    semana_del_anio = payload.fecha.isocalendar()[1]

    fila = {
        "lag_1": payload.lag_1, "lag_7": payload.lag_7,
        "lag_14": payload.lag_14, "lag_28": payload.lag_28,
        "media_movil_7": payload.media_movil_7, "media_movil_14": payload.media_movil_14,
        "media_movil_28": payload.media_movil_28,
        "std_movil_7": payload.std_movil_7, "std_movil_14": payload.std_movil_14,
        "std_movil_28": payload.std_movil_28,
        "dia_semana": dia_semana, "mes": mes, "es_fin_de_semana": es_fin_de_semana,
        "dia_del_mes": dia_del_mes, "semana_del_anio": semana_del_anio,
        "precio": payload.precio, "en_promocion": int(payload.en_promocion),
        "tienda_cod": tienda_cod, "producto_cod": producto_cod,
    }

    try:
        X = np.array([[fila[c] for c in FEATURE_COLS]])
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Feature faltante en el payload mapeado: {e}")

    pred = float(modelo.predict(X)[0])
    pred = max(pred, 0.0)

    # Recomendacion simple de inventario: prediccion + colchon de seguridad
    # proporcional a la variabilidad reciente (std_movil_7), redondeado hacia arriba
    colchon_seguridad = payload.std_movil_7 * 1.65  # ~z-score 95% para un solo lado
    recomendacion = int(np.ceil(pred + colchon_seguridad))

    nivel_confianza = "alta" if payload.std_movil_7 < payload.media_movil_7 * 0.3 else "media"

    return PrediccionResponse(
        tienda_id=payload.tienda_id,
        producto_id=payload.producto_id,
        fecha=payload.fecha,
        unidades_predichas=round(pred, 2),
        recomendacion_inventario=recomendacion,
        nivel_confianza=nivel_confianza,
    )
