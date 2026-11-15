"""Indicadores térmicos: GDD, calor en etapa crítica y noches frescas (v0.7 §9.2, §9.6)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gdd(tmax: pd.Series, tmin: pd.Series, tbase: float, tupper: float | None = None) -> pd.Series:
    """Grados-día diarios: máx(0, (mín(Tmax, T_upper) + Tmin)/2 − Tbase)."""
    tx = tmax.clip(upper=tupper) if tupper is not None else tmax
    return ((tx + tmin) / 2.0 - tbase).clip(lower=0.0)


def dias_sobre_umbral(tmax: pd.Series, umbral: float, mascara: pd.Series | None = None) -> int:
    """Número de días con Tmax > umbral (opcionalmente dentro de una etapa)."""
    sel = tmax if mascara is None else tmax[mascara.reindex(tmax.index).fillna(False)]
    return int((sel > umbral).sum())


def noches_frescas_por_anio(tmin: pd.Series, umbral: float,
                            inicio=(6, 1), fin=(8, 31)) -> pd.Series:
    """Noches con Tmin < umbral entre ``inicio`` y ``fin`` (mes, día) de cada año.

    Solo se cuentan años con la ventana completa (sin días faltantes).
    """
    s = tmin.dropna()
    out = {}
    for anio in sorted(set(s.index.year)):
        a, b = pd.Timestamp(anio, *inicio), pd.Timestamp(anio, *fin)
        ventana = tmin.loc[a:b]
        if len(ventana) == (b - a).days + 1 and ventana.notna().all():
            out[anio] = int((ventana < umbral).sum())
    return pd.Series(out, dtype="int64", name="noches_frescas")


def tmin_media_invierno(tmin: pd.Series, inicio=(6, 1), fin=(8, 31)) -> pd.Series:
    """Tmin media de la ventana de inducción por año (°C), solo años completos.

    Complemento continuo del conteo de noches frescas: con series regionales (POWER, ~50 km)
    las mínimas se suavizan y el conteo bajo un umbral fijo puede salir 0 casi todos los años
    (resultado degenerado, EX-13). La media no depende del umbral y sí ordena los inviernos.
    """
    out = {}
    for anio in sorted(set(tmin.dropna().index.year)):
        a, b = pd.Timestamp(anio, *inicio), pd.Timestamp(anio, *fin)
        v = tmin.loc[a:b]
        if len(v) == (b - a).days + 1 and v.notna().all():
            out[anio] = float(v.mean())
    return pd.Series(out, dtype="float64", name="tmin_media_invierno")


def conteo_degenerado(conteos: pd.Series, minimo_distintos: int = 3) -> bool:
    """``True`` si el conteo no discrimina entre años (p. ej. casi todos 0)."""
    return conteos.nunique() < minimo_distintos or float((conteos == 0).mean()) >= 0.8


def noches_frescas_a_fecha(tmin: pd.Series, umbral: float, fecha: str | pd.Timestamp,
                           inicio=(6, 1)) -> dict:
    """Conteo del invierno en curso hasta ``fecha`` comparado con años anteriores a esa fecha.

    Devuelve el conteo actual, la mediana histórica y el percentil del año actual entre los
    años completos disponibles (categoría: modelado para el conteo, escenario para la
    comparación).
    """
    fecha = pd.Timestamp(fecha).normalize()
    a = pd.Timestamp(fecha.year, *inicio)
    if fecha < a:
        raise ValueError("La fecha es anterior al inicio de la ventana de inducción")
    actual = int((tmin.loc[a:fecha] < umbral).sum())
    historico = []
    for anio in sorted(set(tmin.index.year) - {fecha.year}):
        a0 = pd.Timestamp(anio, *inicio)
        f0 = a0 + (fecha - a)
        v = tmin.loc[a0:f0]
        if len(v) == (f0 - a0).days + 1 and v.notna().all():
            historico.append(int((v < umbral).sum()))
    hist = np.array(historico)
    percentil = float((hist < actual).mean() * 100 + (hist == actual).mean() * 50) if len(hist) else None
    return {
        "actual": actual,
        "mediana_historica": float(np.median(hist)) if len(hist) else None,
        "percentil": percentil,
        "anios_comparados": len(hist),
    }
