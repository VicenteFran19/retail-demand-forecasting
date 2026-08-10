import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "num_features" in body
    assert "features_esperadas" in body


def test_predict_endpoint_devuelve_prediccion_valida():
    payload = {
        "tienda_id": "TIENDA_01",
        "producto_id": "SKU_0001",
        "fecha": "2026-01-15",
        "lag_1": 25.0, "lag_7": 22.0, "lag_14": 20.0, "lag_28": 18.0,
        "media_movil_7": 23.5, "media_movil_14": 21.0, "media_movil_28": 19.5,
        "std_movil_7": 3.2, "std_movil_14": 4.1, "std_movil_28": 5.0,
        "precio": 45.0, "en_promocion": False,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["unidades_predichas"] >= 0
    assert body["recomendacion_inventario"] >= body["unidades_predichas"]
    assert body["nivel_confianza"] in {"alta", "media"}


def test_predict_endpoint_rechaza_payload_incompleto():
    payload = {"tienda_id": "TIENDA_01"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # error de validacion de Pydantic
