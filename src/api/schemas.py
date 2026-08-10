from pydantic import BaseModel, Field
from datetime import date


class PrediccionRequest(BaseModel):
    tienda_id: str = Field(..., examples=["TIENDA_01"])
    producto_id: str = Field(..., examples=["SKU_0001"])
    fecha: date = Field(..., examples=["2026-01-15"])
    lag_1: float = Field(..., description="Unidades vendidas el dia anterior")
    lag_7: float
    lag_14: float
    lag_28: float
    media_movil_7: float
    media_movil_14: float
    media_movil_28: float
    std_movil_7: float
    std_movil_14: float
    std_movil_28: float
    precio: float
    en_promocion: bool = False


class PrediccionResponse(BaseModel):
    tienda_id: str
    producto_id: str
    fecha: date
    unidades_predichas: float
    recomendacion_inventario: int
    nivel_confianza: str
