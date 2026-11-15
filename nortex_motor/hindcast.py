"""Hindcast de escenarios (v0.7 §6.5): ¿el rango P10–P90 habría contenido lo observado?

Para cada año ``t`` con al menos ``min_anios`` años previos, los percentiles se calculan solo
con años anteriores a ``t`` y se comprueba si la métrica observada en ``t`` cayó dentro del
rango. Por construcción se espera ~80 %; la v0.7 exige al menos 7 de cada 10. Con pocos años
el test discrimina poco, así que se reporta el intervalo de confianza de Wilson al 95 %.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nortex_motor.escenarios import ventanas_historicas

CRITERIO_V07 = 0.7


def wilson(aciertos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza de Wilson para una proporción."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = aciertos / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centro - margen), min(1.0, centro + margen))


def hindcast(clima: pd.DataFrame, inicio_mm_dd: str, dias: int, metrica: str = "deficit",
             min_anios: int = 10, bajo: int = 10, alto: int = 90,
             anios: list[int] | None = None) -> dict:
    """Prueba retrospectiva del rango de escenarios para una ventana de campaña."""
    tabla = ventanas_historicas(clima, inicio_mm_dd, dias, anios)
    if tabla.empty:
        raise ValueError("Sin ventanas completas para el hindcast")
    filas = []
    for anio in tabla.index:
        previos = tabla.loc[tabla.index < anio, metrica]
        if len(previos) < min_anios:
            continue
        p_bajo, p_alto = np.percentile(previos, [bajo, alto])
        obs = float(tabla.loc[anio, metrica])
        filas.append({"anio": int(anio), "observado": obs, f"p{bajo}": float(p_bajo),
                      f"p{alto}": float(p_alto), "dentro": bool(p_bajo <= obs <= p_alto)})
    detalle = pd.DataFrame(filas)
    n = len(detalle)
    aciertos = int(detalle["dentro"].sum()) if n else 0
    cobertura = aciertos / n if n else float("nan")
    ic = wilson(aciertos, n)
    if n == 0:
        veredicto = "sin años suficientes"
    elif cobertura >= CRITERIO_V07:
        veredicto = "calibrado (≥ 7 de 10)"
    else:
        veredicto = "mal calibrado (< 7 de 10)"
    return {"metrica": metrica, "ventana": f"{inicio_mm_dd} + {dias} días", "anios_probados": n,
            "aciertos": aciertos, "cobertura": cobertura, "ic95": ic,
            "esperado": (alto - bajo) / 100, "veredicto": veredicto, "detalle": detalle}
