"""Situación del reservorio San Lorenzo (D-17) y dotación análoga (EX-15).

Entradas: serie de volumen almacenado (hm³) de ANA, de la Junta o de los reportes
hidrológicos de SENAMHI, y opcionalmente el historial de PCR (dotación por campaña).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CAPACIDAD_UTIL_HM3 = 195.6   # SENAMHI 2022–2026 (reducida por sedimentación)
MIN_ANIOS_PERCENTIL = 5


def serie_volumen(df: pd.DataFrame, reservorio: str = "san_lorenzo") -> pd.Series:
    """Serie diaria de volumen (hm³) desde el formato largo de nortex_data.

    Acepta ``reservoir_volume`` (plantilla local, ANA) y ``reservoir_volume_report``
    (reportes SENAMHI, filtrando el reservorio por ``product``). Si hay dos valores el mismo
    día prevalece el local.
    """
    rep = df[(df["variable"] == "reservoir_volume_report")
             & df["product"].str.contains(reservorio, case=False, na=False)]
    loc = df[df["variable"] == "reservoir_volume"]
    serie = pd.concat([rep.set_index("date")["value"], loc.set_index("date")["value"]])
    serie.index = pd.to_datetime(serie.index)
    return serie.groupby(level=0).last().sort_index().dropna()


def estado(volumen: pd.Series, fecha: str | pd.Timestamp,
           capacidad_util: float = CAPACIDAD_UTIL_HM3, ventana_dias: int = 15) -> dict:
    """Volumen a la fecha, % de capacidad útil y percentil frente a otros años en la misma época.

    El percentil compara el volumen actual con la media de cada año histórico en ±``ventana_dias``
    alrededor del mismo día del año. Con menos de 5 años se informa «historia insuficiente».
    Categoría: «bajo» (< P25), «normal», «alto» (> P75).
    """
    fecha = pd.Timestamp(fecha).normalize()
    previo = volumen.loc[:fecha]
    if previo.empty:
        return {"volumen_hm3": None, "fecha_dato": None, "nota": "sin datos hasta la fecha"}
    fecha_dato, actual = previo.index[-1], float(previo.iloc[-1])
    out = {"volumen_hm3": actual, "fecha_dato": fecha_dato.date().isoformat(),
           "antiguedad_dias": int((fecha - fecha_dato).days),
           "porcentaje_capacidad": round(100 * actual / capacidad_util, 1),
           "capacidad_util_hm3": capacidad_util}
    doy = fecha_dato.day_of_year
    hist = []
    for anio, g in volumen.groupby(volumen.index.year):
        if anio == fecha_dato.year:
            continue
        dist = np.abs(((g.index.day_of_year - doy) + 182) % 365 - 182)
        cerca = g[dist <= ventana_dias]
        if len(cerca):
            hist.append(float(cerca.mean()))
    out["anios_comparados"] = len(hist)
    if len(hist) < MIN_ANIOS_PERCENTIL:
        out.update(percentil=None, categoria="por determinar",
                   nota=f"historia insuficiente ({len(hist)} años; se requieren {MIN_ANIOS_PERCENTIL})")
        return out
    h = np.array(hist)
    pct = float((h < actual).mean() * 100 + (h == actual).mean() * 50)
    out["percentil"] = round(pct, 1)
    out["categoria"] = "bajo" if pct < 25 else "alto" if pct > 75 else "normal"
    return out


def dotacion_analoga(pcr: pd.DataFrame, volumen: pd.Series, volumen_actual: float,
                     cultivo: str, max_desfase_dias: int = 30) -> dict | None:
    """Dotación de la campaña histórica con volumen inicial más parecido (EX-15).

    ``pcr`` tiene columnas ``fecha_inicio, cultivo, dotacion_m3_ha``. Devuelve la dotación y la
    campaña análoga (categoría «escenario»), o ``None`` si no hay campañas comparables.
    """
    candidatos = []
    for _, fila in pcr[pcr["cultivo"].str.lower() == cultivo.lower()].iterrows():
        inicio = pd.Timestamp(fila["fecha_inicio"])
        previo = volumen.loc[inicio - pd.Timedelta(days=max_desfase_dias):inicio]
        if len(previo):
            candidatos.append((abs(float(previo.iloc[-1]) - volumen_actual), inicio,
                               float(fila["dotacion_m3_ha"]), float(previo.iloc[-1])))
    if not candidatos:
        return None
    _, inicio, dot, vol = min(candidatos)
    return {"dotacion_m3_ha": dot, "campania_analoga": inicio.date().isoformat(),
            "volumen_inicial_analogo_hm3": vol, "categoria": "escenario"}
