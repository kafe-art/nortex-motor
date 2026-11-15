"""Adaptador de clima: convierte la salida larga de `nortex_data` en una tabla diaria ancha.

Acepta los dos juegos de ids que existen hoy en el equipo (p. ej. ``precip_power`` del
repositorio kafe-art y ``rain_power`` del de BRYTNEI), para no depender de esa decisión.
"""

from __future__ import annotations

import pandas as pd

#: Columna interna → ids aceptados en la salida de nortex_data (en orden de preferencia).
ALIAS = {
    "tmax": ("tmax",),
    "tmin": ("tmin",),
    "et0": ("et0",),
    "precip": ("precip_power", "rain_power", "precipitation", "precip"),
}
REQUERIDAS = ("tmax", "tmin", "et0", "precip")


def desde_largo(df: pd.DataFrame, fuente_precip: str | None = None) -> pd.DataFrame:
    """Pasa del esquema largo (``date, variable, value``) a columnas ``tmax, tmin, et0, precip``.

    ``fuente_precip`` fuerza un id concreto de precipitación (p. ej. ``"precipitation"`` de
    IMERG en lugar de la de POWER). Lanza ``KeyError`` si falta alguna variable requerida.
    """
    ancho = df.pivot_table(index="date", columns="variable", values="value", aggfunc="first")
    ancho.index = pd.to_datetime(ancho.index)
    out = {}
    for col, alias in ALIAS.items():
        opciones = (fuente_precip,) if (col == "precip" and fuente_precip) else alias
        elegido = next((a for a in opciones if a in ancho.columns), None)
        if elegido is None:
            raise KeyError(f"Falta {col}: se buscó {opciones} en {sorted(ancho.columns)}")
        out[col] = ancho[elegido]
    return pd.DataFrame(out).sort_index().asfreq("D")


def validar(clima: pd.DataFrame) -> None:
    """Comprueba columnas y frecuencia diaria de una tabla de clima."""
    faltan = [c for c in REQUERIDAS if c not in clima.columns]
    if faltan:
        raise KeyError(f"La tabla de clima no tiene {faltan}")
    if not isinstance(clima.index, pd.DatetimeIndex):
        raise TypeError("El índice debe ser de fechas")
