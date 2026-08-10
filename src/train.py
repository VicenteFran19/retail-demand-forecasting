"""
Entrena el modelo LightGBM de forecasting de demanda, compara contra el
baseline estacional, y registra todo en MLflow (parametros, metricas y
el modelo serializado).
"""

import json
import sys
import os

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import construir_features, split_temporal, FEATURE_COLS, TARGET_COL
from src.baseline import prediccion_baseline_estacional
from src.evaluate import resumen_metricas


def main():
    print("1. Cargando datos...")
    df = pd.read_parquet("data/ventas_sinteticas.parquet")

    print("2. Construyendo features (lags, ventanas moviles, calendario)...")
    df_feat = construir_features(df)

    print("3. Split temporal (train / val / test = 28 dias con margen de 3 dias)...")
    train, val, test = split_temporal(df_feat, dias_test=28, dias_val=28, margen_dias=3)
    print(f"   Train: {train['fecha'].min().date()} a {train['fecha'].max().date()} ({len(train):,} filas)")
    print(f"   Val:   {val['fecha'].min().date()} a {val['fecha'].max().date()} ({len(val):,} filas)")
    print(f"   Test:  {test['fecha'].min().date()} a {test['fecha'].max().date()} ({len(test):,} filas)")

    # Quitamos filas con NaN en features criticas (los primeros 28 dias de cada serie
    # no tienen suficiente historia para la ventana movil de 28 dias)
    cols_necesarias = FEATURE_COLS + [TARGET_COL]
    train = train.dropna(subset=cols_necesarias)
    val = val.dropna(subset=cols_necesarias)
    test = test.dropna(subset=cols_necesarias)

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_val, y_val = val[FEATURE_COLS], val[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    print("\n4. Calculando baseline estacional sobre test...")
    pred_baseline_test = prediccion_baseline_estacional(
        pd.concat([train, val, test]).sort_values(["tienda_id", "producto_id", "fecha"])
    ).loc[test.index]
    metricas_baseline = resumen_metricas(y_test.values, pred_baseline_test.values)
    print(f"   Baseline WMAPE: {metricas_baseline['WMAPE_%']}%")

    print("\n5. Entrenando LightGBM...")
    mlflow.set_experiment("retail-demand-forecasting")

    params = {
        "objective": "regression",
        "metric": "mae",
        "num_leaves": 63,
        "max_depth": -1,          # dejamos que num_leaves controle la complejidad, sin conflicto
        "learning_rate": 0.05,
        "n_estimators": 800,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "verbosity": -1,
    }

    with mlflow.start_run(run_name="lightgbm_forecast_v1"):
        mlflow.log_params(params)

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        pred_test = model.predict(X_test)
        metricas_modelo = resumen_metricas(y_test.values, pred_test)

        mejora_relativa = (
            (metricas_baseline["WMAPE_%"] - metricas_modelo["WMAPE_%"])
            / metricas_baseline["WMAPE_%"] * 100
        )

        print(f"\n   Modelo LightGBM WMAPE: {metricas_modelo['WMAPE_%']}%")
        print(f"   Baseline estacional WMAPE: {metricas_baseline['WMAPE_%']}%")
        print(f"   Mejora relativa: {mejora_relativa:.2f}%")

        mlflow.log_metrics({
            "wmape_modelo": metricas_modelo["WMAPE_%"],
            "wmape_baseline": metricas_baseline["WMAPE_%"],
            "mejora_relativa_pct": mejora_relativa,
            "mae_modelo": metricas_modelo["MAE"],
            "rmse_modelo": metricas_modelo["RMSE"],
            "best_iteration": model.best_iteration_ or params["n_estimators"],
        })

        mlflow.lightgbm.log_model(model, artifact_path="model")

        # Guardamos tambien localmente para que la API lo pueda cargar sin depender de MLflow tracking server
        os.makedirs("models", exist_ok=True)
        model.booster_.save_model("models/lightgbm_model.txt")

        importancias = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        print("\n6. Top 10 features mas importantes:")
        print(importancias.head(10).to_string())

        resultado = {
            "wmape_modelo": metricas_modelo["WMAPE_%"],
            "wmape_baseline": metricas_baseline["WMAPE_%"],
            "mejora_relativa_pct": round(mejora_relativa, 2),
            "mae_modelo": metricas_modelo["MAE"],
            "rmse_modelo": metricas_modelo["RMSE"],
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        }
        with open("models/metricas_finales.json", "w") as f:
            json.dump(resultado, f, indent=2)

        print("\n7. Modelo y metricas guardados en models/")
        print(json.dumps(resultado, indent=2))

    return resultado


if __name__ == "__main__":
    main()
