# Retail Demand Forecasting & Inventory Optimizer

Solución end-to-end para pronosticar demanda diaria por tienda y producto, y generar recomendaciones de inventario, sobre dos años de datos sintéticos reproducibles.

## Resultados

| Métrica | Baseline estacional | LightGBM |
|---|---|---|
| WMAPE | 15.42% | **11.32%** |
| Mejora relativa | — | **26.54%** |

*Métricas obtenidas sobre un conjunto de test de 28 días, con split temporal estricto y margen de 3 días entre conjuntos para evitar fuga de información. Reproducibles corriendo `python src/train.py` con `seed=42`.*

## Arquitectura

```
data/generate_data.py   -> genera dataset sintético reproducible (2 años, 8 tiendas, 25 productos)
src/features.py         -> feature engineering: lags, ventanas móviles, calendario (sin leakage)
src/baseline.py         -> baseline estacional naive (punto de comparación)
src/train.py            -> entrenamiento LightGBM + tracking en MLflow
src/evaluate.py         -> métricas: WMAPE, MAE, RMSE
src/monitoring/psi.py   -> detección de drift con Population Stability Index
src/api/main.py         -> API FastAPI que sirve el modelo
src/api/schemas.py      -> contratos de entrada/salida con Pydantic
tests/                  -> pruebas automatizadas (feature engineering + API)
```

## Cómo evitamos fuga de información (data leakage)

1. **Todo lag y ventana móvil usa `shift(1)` antes de calcular cualquier agregación** — la ventana del día D nunca incluye el día D.
2. **Split temporal por fecha completa, no por posición de fila**, con 3 días de margen entre train/val/test, para que ninguna ventana móvil de un conjunto se calcule mezclando datos del borde del otro.
3. **Tests automatizados específicos** (`test_lag_1_no_usa_informacion_del_mismo_dia`, `test_media_movil_no_incluye_dia_actual`) que verifican esto matemáticamente, no solo de forma manual.

## Cómo correrlo

```bash
pip install -r requirements.txt

# 1. Generar los datos sintéticos
python data/generate_data.py

# 2. Entrenar el modelo (registra en MLflow y guarda en models/)
python src/train.py

# 3. Correr los tests
python -m pytest tests/ -v

# 4. Levantar la API
uvicorn src.api.main:app --reload

# 5. Probar una predicción
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tienda_id":"TIENDA_01","producto_id":"SKU_0001","fecha":"2026-01-15","lag_1":25.0,"lag_7":22.0,"lag_14":20.0,"lag_28":18.0,"media_movil_7":23.5,"media_movil_14":21.0,"media_movil_28":19.5,"std_movil_7":3.2,"std_movil_14":4.1,"std_movil_28":5.0,"precio":45.0,"en_promocion":false}'
```

## Con Docker

```bash
docker build -t retail-forecasting .
docker run -p 8000:8000 retail-forecasting
```

## Monitoreo de drift (PSI)

```bash
python src/monitoring/psi.py
```

Compara la distribución de features entre el período de entrenamiento y el de test. Regla de interpretación estándar de la industria:

- PSI < 0.10 → sin cambio significativo
- 0.10 ≤ PSI < 0.25 → cambio moderado, vigilar
- PSI ≥ 0.25 → cambio importante, considerar reentrenar

## Limitaciones conocidas (honestidad técnica)

- Los datos son sintéticos, generados con patrones de estacionalidad y tendencia razonables, pero no capturan toda la complejidad de demanda real (eventos externos, canibalización entre productos, elasticidad-precio real).
- El mapeo de `tienda_id`/`producto_id` a códigos categóricos en la API usa un hash simplificado; en producción esto debería persistirse junto con el modelo (ej. como parte de los artefactos de MLflow) para garantizar consistencia exacta entre entrenamiento y serving.
- El colchón de seguridad de inventario usa una heurística simple (`z ≈ 1.65` sobre `std_movil_7`); un sistema de producción real debería optimizar el nivel de servicio objetivo según el costo de quiebre de stock vs. costo de sobre-inventario por SKU.

## Stack técnico

Python · Pandas · LightGBM · Scikit-learn · FastAPI · Pydantic · MLflow · Docker · Pytest · GitHub Actions · PSI (drift monitoring)
