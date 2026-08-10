"""
Genera datos sinteticos reproducibles de demanda diaria por tienda y producto.

Incluye, a proposito, los patrones que un modelo de forecasting real deberia
poder capturar:
- Estacionalidad semanal (mas ventas en fin de semana)
- Estacionalidad anual (picos en diciembre / campanas)
- Tendencia de crecimiento leve
- Efecto de precio/promocion
- Ruido realista (no todo es predecible)
- Dias sin venta (stockouts ocasionales) -> demanda censurada, reto real de retail
"""

import numpy as np
import pandas as pd

SEED = 42
N_TIENDAS = 8
N_PRODUCTOS = 25
FECHA_INICIO = "2024-01-01"
FECHA_FIN = "2025-12-31"


def generar_dataset(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    fechas = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D")
    tiendas = [f"TIENDA_{i:02d}" for i in range(1, N_TIENDAS + 1)]
    productos = [f"SKU_{i:04d}" for i in range(1, N_PRODUCTOS + 1)]

    # Cada tienda tiene un tamano/nivel base distinto (unas venden mas que otras)
    factor_tienda = {t: rng.uniform(0.6, 1.8) for t in tiendas}
    # Cada producto tiene una demanda base distinta (categoria implicita)
    demanda_base_producto = {p: rng.uniform(5, 60) for p in productos}
    # Cada producto tiene un precio base
    precio_base_producto = {p: round(rng.uniform(8, 120), 2) for p in productos}

    filas = []
    dia_cero = fechas[0]

    for tienda in tiendas:
        for producto in productos:
            base = demanda_base_producto[producto] * factor_tienda[tienda]
            precio_base = precio_base_producto[producto]

            for fecha in fechas:
                dia_index = (fecha - dia_cero).days

                # Tendencia: crecimiento leve a lo largo de 2 anios
                tendencia = 1 + 0.00015 * dia_index

                # Estacionalidad semanal: mas venta viernes/sabado/domingo
                dow = fecha.dayofweek  # 0=lunes ... 6=domingo
                factor_semana = {
                    0: 0.90, 1: 0.88, 2: 0.92, 3: 0.97,
                    4: 1.15, 5: 1.35, 6: 1.10,
                }[dow]

                # Estacionalidad anual: pico en noviembre-diciembre (campanas)
                mes = fecha.month
                factor_mes = {
                    1: 0.85, 2: 0.85, 3: 0.90, 4: 0.95, 5: 1.00, 6: 1.00,
                    7: 1.05, 8: 1.00, 9: 0.95, 10: 1.05, 11: 1.30, 12: 1.45,
                }[mes]

                # Promocion aleatoria: ~8% de los dias hay descuento, sube demanda
                en_promocion = rng.random() < 0.08
                descuento_pct = rng.uniform(0.10, 0.30) if en_promocion else 0.0
                precio_dia = round(precio_base * (1 - descuento_pct), 2)
                factor_promo = 1 + (descuento_pct * 1.8) if en_promocion else 1.0

                media_esperada = base * tendencia * factor_semana * factor_mes * factor_promo
                media_esperada = max(media_esperada, 0.5)

                # Ruido tipo Poisson/NegBin (conteos, no negativos)
                unidades_vendidas = rng.poisson(lam=media_esperada)

                # Stockout ocasional: ~2% de los dias, no hay stock suficiente
                # y la venta observada queda censurada (demanda real > venta registrada)
                hubo_quiebre_stock = rng.random() < 0.02
                if hubo_quiebre_stock:
                    unidades_vendidas = int(unidades_vendidas * rng.uniform(0.0, 0.4))

                filas.append((
                    fecha, tienda, producto, unidades_vendidas,
                    precio_dia, en_promocion, hubo_quiebre_stock,
                ))

    df = pd.DataFrame(
        filas,
        columns=[
            "fecha", "tienda_id", "producto_id", "unidades_vendidas",
            "precio", "en_promocion", "hubo_quiebre_stock",
        ],
    )
    df = df.sort_values(["tienda_id", "producto_id", "fecha"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generar_dataset()
    out_path = "data/ventas_sinteticas.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Dataset generado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    print(f"Rango de fechas: {df['fecha'].min().date()} a {df['fecha'].max().date()}")
    print(f"Tiendas: {df['tienda_id'].nunique()} | Productos: {df['producto_id'].nunique()}")
    print(f"Guardado en: {out_path}")
