"""Riego por turnos: calendario, cobertura hídrica (v0.7 §9.4) y autonomía (§9.5)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Eficiencias de aplicación de referencia (POR VALIDAR, v0.7 V-33).
EFICIENCIA_REFERENCIA = {"gravedad": 0.5, "goteo": 0.9}


@dataclass(frozen=True)
class Riego:
    """Condiciones de riego declaradas por el agricultor o la Junta.

    ``intervalo_dias``: días entre turnos. ``dotacion_m3_ha``: volumen asignado por el PCR
    para toda la campaña (``None`` = no confirmado → resultados «por determinar», EX-04).
    ``eficiencia``: fracción del agua bruta que llega a la zona radicular.
    ``modo``: ``"dotacion"`` reparte la dotación en láminas iguales por turno;
    ``"llenado"`` repone hasta capacidad de campo en cada turno (agua no limitante).
    ``primer_turno``: día de la campaña del primer turno (0 = el día de siembra).
    """

    intervalo_dias: int | None
    dotacion_m3_ha: float | None = None
    eficiencia: float = EFICIENCIA_REFERENCIA["gravedad"]
    modo: str = "dotacion"
    primer_turno: int = 0

    def __post_init__(self) -> None:
        if self.intervalo_dias is not None and self.intervalo_dias < 1:
            raise ValueError("intervalo_dias debe ser ≥ 1")
        if not 0 < self.eficiencia <= 1:
            raise ValueError("eficiencia debe estar en (0, 1]")
        if self.modo not in {"dotacion", "llenado"}:
            raise ValueError("modo debe ser 'dotacion' o 'llenado'")


def dias_de_turno(fechas: pd.DatetimeIndex, riego: Riego) -> pd.Series:
    """Serie booleana con los días de turno dentro de la campaña."""
    flags = pd.Series(False, index=fechas)
    if riego.intervalo_dias:
        flags.iloc[riego.primer_turno::riego.intervalo_dias] = True
    return flags


def lamina_por_turno(fechas: pd.DatetimeIndex, riego: Riego) -> pd.Series:
    """Lámina NETA por día (mm) repartiendo la dotación en turnos iguales.

    1 m³/ha = 0,1 mm, así que bruta por turno = dotación / 10 / n_turnos y neta = bruta · Ea.
    """
    turnos = dias_de_turno(fechas, riego)
    n = int(turnos.sum())
    if riego.dotacion_m3_ha is None or n == 0:
        return pd.Series(0.0, index=fechas)
    neta = riego.dotacion_m3_ha / 10.0 / n * riego.eficiencia
    return turnos.astype(float) * neta


def cobertura_hidrica(demanda_neta_mm: float, eficiencia: float, area_ha: float,
                      dotacion_m3_ha: float | None) -> dict:
    """Cobertura C = oferta asignada / demanda bruta, ambas en m³ (v0.7 §9.4).

    Devuelve ``demanda_bruta_m3``, ``oferta_m3``, ``c`` y ``etiqueta``
    (Cubierto si C ≥ 1; No cubierto si C < 1; Por determinar sin dotación).
    """
    db_m3 = demanda_neta_mm / eficiencia * 10.0 * area_ha
    if dotacion_m3_ha is None:
        return {"demanda_bruta_m3": db_m3, "oferta_m3": None, "c": None,
                "etiqueta": "Por determinar"}
    oferta = dotacion_m3_ha * area_ha
    c = oferta / db_m3 if db_m3 > 0 else float("inf")
    return {"demanda_bruta_m3": db_m3, "oferta_m3": oferta, "c": c,
            "etiqueta": "Cubierto" if c >= 1 else "No cubierto"}


def autonomia_dias(raw_mm: float, etc_mm_dia: float) -> float:
    """Días que la RAW cubre la demanda: RAW / ETc (v0.7 §9.5)."""
    if etc_mm_dia <= 0:
        return float("inf")
    return raw_mm / etc_mm_dia


def dias_estres_por_turno(intervalo_dias: int | None, autonomia: float) -> float | None:
    """Días de estrés previstos por turno: máx(0, intervalo − autonomía)."""
    if intervalo_dias is None:
        return None
    return max(0.0, intervalo_dias - autonomia)
