"""Riesgo climático por cultivo: exceso de lluvia y calor en floración (frecuencia histórica).

Los escenarios P10/P50/P90 se ordenan por déficit de agua, así que el año más lluvioso o más
caluroso puede no estar entre ellos. Para el exceso de lluvia y el calor se usa la **frecuencia
en todos los años del registro** y el **peor año** como ejemplo (v0.7 §6.6: «solo frecuencia
histórica»), y además se describe el año análogo de cada escenario.

- **Exceso de lluvia (potencial):** RX5day (máxima lluvia en 5 días) dentro de la campaña por
  encima de un umbral **relativo y local**: el aguacero de 5 días que se da **1 vez cada 5 años**
  (P80 de los máximos anuales del registro completo). Así la frecuencia depende de la época de
  siembra y no puede pasar de ~20 % de los años salvo en registros con muchos Niños.
  (Se descartó el P90 de todas las ventanas húmedas: en San Lorenzo da ~18 mm y se supera en el
  60 % de los años, por lo que todo salía «alto».)
  El nivel se ajusta por la **sensibilidad del cultivo** (ficha) y luego, en `decision`, por el
  **drenaje** (pendiente, arcilla, napa). Lluvia intensa ≠ encharcamiento confirmado.
- **Calor en floración:** días con Tmax sobre el umbral del cultivo **dentro de su ventana de
  floración** (días después de la siembra en anuales; meses en perennes). Un año cuenta si tiene
  al menos `MIN_DIAS_CALOR` días así. Usar el clima **corregido con la estación**: POWER crudo
  sobrestima la Tmax de San Lorenzo en ~2,7 °C.

Fuente de lluvia: IMERG (0,1°, capta mejor los aguaceros) si cubre ≥ 15 años; si no, POWER.
Todos los cortes están declarados aquí y son POR VALIDAR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nortex_motor.catalogo import cultivo as get_cultivo

PERIODO_RETORNO_ANIOS = 5      # umbral = aguacero de 5 días que se da 1 vez cada 5 años
MIN_ANIOS_IMERG = 15
MIN_DIAS_CALOR = 3              # días con Tmax > umbral en floración para contar el año
#: Frecuencia de años con exceso → nivel, según sensibilidad: (corte medio, corte alto).
CORTES_EXCESO = {"alta": (0.05, 0.15), "media": (0.10, 0.20), "baja": (0.15, 0.30)}
#: Frecuencia de años con calor en floración → nivel.
CORTES_CALOR = (0.10, 0.30)


def umbral_lluvia(precip: pd.Series) -> float | None:
    """Aguacero de 5 días con periodo de retorno de 5 años (P80 de los máximos anuales)."""
    suma5 = precip.rolling(5).sum()
    anios = suma5.groupby(suma5.index.year).agg(["max", "count"])
    maximos = anios.loc[anios["count"] >= 330, "max"].dropna()  # solo años casi completos
    if len(maximos) < 5:
        return None
    return float(np.percentile(maximos, 100 * (1 - 1 / PERIODO_RETORNO_ANIOS)))


def _nivel(frecuencia: float, cortes: tuple[float, float]) -> str:
    medio, alto = cortes
    return "alto" if frecuencia >= alto else "medio" if frecuencia >= medio else "bajo"


def mascara_floracion(cultivo_id: str, fechas: pd.DatetimeIndex) -> pd.Series | None:
    """Días de la campaña que caen en la ventana de floración (``None`` si no hay dato)."""
    c = get_cultivo(cultivo_id)
    if c.perenne:
        meses = c.get("floracion_meses")
        return None if not meses else pd.Series(fechas.month.isin(meses), index=fechas)
    rango = c.get("floracion_dias")
    if not rango:
        return None
    dia = np.arange(len(fechas))
    return pd.Series((dia >= rango[0]) & (dia <= rango[1]), index=fechas)


def describir_ventana(cultivo_id: str) -> str:
    c = get_cultivo(cultivo_id)
    if c.perenne:
        meses = c.get("floracion_meses")
        return "sin dato de floración" if not meses else f"meses {meses}"
    a, b = c.get("floracion_dias") or (None, None)
    return "sin dato de floración" if a is None else f"días {a}–{b} después de la siembra"


def evaluar(cultivo_id: str, inicio, clima: pd.DataFrame, lluvia: pd.Series | None = None) -> dict:
    """Frecuencia histórica de exceso de lluvia y de calor en floración para una fecha de siembra."""
    c = get_cultivo(cultivo_id)
    inicio = pd.Timestamp(inicio)
    dias = c.ciclo_dias or 365
    usar_imerg = lluvia is not None and len(set(lluvia.dropna().index.year)) >= MIN_ANIOS_IMERG
    precip = lluvia.dropna() if usar_imerg else clima["precip"].dropna()
    fuente = "IMERG (0,1°)" if usar_imerg else "NASA POWER (celda ~50 km; suaviza los aguaceros)"
    umbral = umbral_lluvia(precip)
    umbral_calor = c.get("calor_floracion_c")
    filas = []
    for anio in sorted(set(precip.index.year) & set(clima["tmax"].dropna().index.year)):
        try:
            a = pd.Timestamp(anio, inicio.month, inicio.day)
        except ValueError:
            a = pd.Timestamp(anio, inicio.month, 28)
        b = a + pd.Timedelta(days=dias - 1)
        p = precip.loc[a:b]
        tx = clima["tmax"].loc[a:b]
        if len(p) < dias * 0.95 or len(tx) < dias * 0.95:
            continue
        rx5 = float(p.rolling(5).sum().max())
        mascara = mascara_floracion(cultivo_id, tx.index)
        calor = (None if mascara is None or umbral_calor is None
                 else int((tx[mascara.to_numpy()] > umbral_calor).sum()))
        filas.append({"anio": anio, "rx5day_mm": round(rx5, 1), "lluvia_total_mm": round(float(p.sum()), 1),
                      "exceso": bool(umbral is not None and rx5 > umbral), "dias_calor_floracion": calor})
    detalle = pd.DataFrame(filas).set_index("anio") if filas else pd.DataFrame()
    out = {"fuente_lluvia": fuente, "umbral_rx5day_mm": None if umbral is None else round(umbral, 1),
           "anios": len(detalle), "ventana_floracion": describir_ventana(cultivo_id),
           "sensibilidad_anegamiento": c.get("sensibilidad_anegamiento"),
           "umbral_calor_c": umbral_calor, "detalle": detalle}
    if detalle.empty:
        out.update(frecuencia_exceso=None, nivel_exceso="por determinar",
                   frecuencia_calor=None, nivel_calor="por determinar")
        return out

    n_exceso = int(detalle["exceso"].sum())
    out["frecuencia_exceso"] = n_exceso / len(detalle)
    out["anios_exceso"] = n_exceso
    peor = detalle["rx5day_mm"].idxmax()
    out["peor_anio_lluvia"] = {"anio": int(peor), "rx5day_mm": float(detalle.loc[peor, "rx5day_mm"])}
    sens = c.get("sensibilidad_anegamiento") or "media"
    out["nivel_exceso"] = _nivel(out["frecuencia_exceso"], CORTES_EXCESO[sens])

    calor = detalle["dias_calor_floracion"]
    if calor.isna().all():
        out.update(frecuencia_calor=None, nivel_calor="por determinar")
    else:
        con_calor = int((calor >= MIN_DIAS_CALOR).sum())
        out["frecuencia_calor"] = con_calor / len(detalle)
        out["anios_calor"] = con_calor
        peor_c = calor.astype(float).idxmax()
        out["peor_anio_calor"] = {"anio": int(peor_c), "dias": int(calor.loc[peor_c])}
        out["nivel_calor"] = _nivel(out["frecuencia_calor"], CORTES_CALOR)
    return out


def clima_por_escenario(extremos: dict, escenarios: list) -> list[dict]:
    """Tres resultados independientes para el año análogo de cada escenario (sin sumarlos)."""
    detalle = extremos.get("detalle")
    out = []
    for e in escenarios:
        fila = (detalle.loc[e.anio_analogo] if detalle is not None and not detalle.empty
                and e.anio_analogo in detalle.index else None)
        out.append({
            "escenario": e.nombre,
            "anio_analogo": e.anio_analogo,
            "deficit_mm": round(e.valor_analogo, 1),
            "nivel_deficit": {10: "bajo", 50: "medio", 90: "alto"}.get(e.percentil, f"P{e.percentil}"),
            "rx5day_mm": None if fila is None else float(fila["rx5day_mm"]),
            "exceso_lluvia": None if fila is None else bool(fila["exceso"]),
            "dias_calor_floracion": (None if fila is None or pd.isna(fila["dias_calor_floracion"])
                                     else int(fila["dias_calor_floracion"])),
        })
    return out
