"""Escenarios por percentil y año análogo (v0.7 §6.2–6.3).

Procedimiento:
1. Para cada año histórico se toma la misma ventana de la campaña (fecha de inicio y duración).
2. Se calcula una métrica por ventana. Por defecto el **déficit climático** Σ(ET0 − P) (mm)
   entre la siembra y el fin del ciclo:
   en un valle regado la lluvia es casi nula y la demanda atmosférica es lo que varía; con la
   precipitación sola los percentiles quedan degenerados (EX-13). La métrica es configurable.
3. Se toman los percentiles 10, 50 y 90 y el **año análogo** más cercano a cada uno. Se nombran
   por el clima que describen (``deficit_bajo``, ``deficit_medio``, ``deficit_alto``), **no** como
   favorable o desfavorable: un año de poco déficit puede traer exceso de lluvia (El Niño) y ser
   malo para un cultivo sensible al anegamiento. La conveniencia se evalúa después, por cultivo.
4. El escenario es la serie diaria real del año análogo, re-fechada a la campaña futura.

Lo que NO se hace: no se asigna probabilidad, no se presenta como pronóstico y no se
extrapolan tendencias. Las salidas llevan categoría «escenario».
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nortex_motor.clima import validar

MIN_ANIOS_ROBUSTO = 15  # EX-06: con menos años se degrada la confianza
PERCENTILES = (10, 50, 90)
#: Qué mide cada métrica, para las etiquetas.
DESCRIPCION_METRICA = {
    "deficit": "déficit climático Σ(ET0 − lluvia) entre la siembra y el fin del ciclo",
    "precip": "lluvia acumulada entre la siembra y el fin del ciclo",
    "et0": "ET0 acumulada entre la siembra y el fin del ciclo",
    "tmax_media": "Tmax media entre la siembra y el fin del ciclo",
}
#: Nombres por percentil: describen el clima, no la conveniencia para el cultivo.
NOMBRES = {
    "deficit": {10: "deficit_bajo", 50: "deficit_medio", 90: "deficit_alto"},
    "precip": {10: "lluvia_baja", 50: "lluvia_media", 90: "lluvia_alta"},
    "et0": {10: "et0_baja", 50: "et0_media", 90: "et0_alta"},
    "tmax_media": {10: "tmax_baja", 50: "tmax_media", 90: "tmax_alta"},
}
METRICAS = {
    "deficit": lambda v: float((v["et0"] - v["precip"]).sum()),
    "precip": lambda v: float(v["precip"].sum()),
    "et0": lambda v: float(v["et0"].sum()),
    "tmax_media": lambda v: float(v["tmax"].mean()),
}


@dataclass(frozen=True)
class Escenario:
    """Escenario construido a partir de un año análogo."""

    nombre: str
    percentil: int
    metrica: str
    valor_percentil: float
    anio_analogo: int
    valor_analogo: float

    @property
    def etiqueta(self) -> str:
        return (f"{self.nombre.replace('_', ' ')}: percentil {self.percentil} de "
                f"{DESCRIPCION_METRICA.get(self.metrica, self.metrica)} (año análogo {self.anio_analogo})")

    @property
    def nota(self) -> str:
        """Aclaración obligatoria en la interfaz (v0.7 §6.2 y §6.7)."""
        return ("Escenario construido con años históricos: describe el clima, no si es bueno o malo "
                "para el cultivo; un percentil no es una probabilidad.")


def ventanas_historicas(clima: pd.DataFrame, inicio_mm_dd: str, dias: int,
                        anios: list[int] | None = None) -> pd.DataFrame:
    """Métricas de la ventana de campaña para cada año con datos completos.

    ``inicio_mm_dd`` es la fecha de inicio de la campaña (``"MM-DD"``) y ``dias`` su duración.
    ``anios`` permite estratificar (p. ej. solo años neutros según el ICEN).
    """
    validar(clima)
    mes, dia = (int(x) for x in inicio_mm_dd.split("-"))
    filas = []
    for anio in sorted(set(clima.index.year)):
        if anios is not None and anio not in anios:
            continue
        try:
            a = pd.Timestamp(anio, mes, dia)
        except ValueError:  # 29 de febrero en año no bisiesto
            a = pd.Timestamp(anio, mes, 28)
        b = a + pd.Timedelta(days=dias - 1)
        v = clima.loc[a:b, ["tmax", "tmin", "et0", "precip"]]
        if len(v) < dias or v.isna().any().any():
            continue
        fila = {"anio": anio, "inicio": a}
        fila.update({k: f(v) for k, f in METRICAS.items()})
        filas.append(fila)
    return pd.DataFrame(filas).set_index("anio") if filas else pd.DataFrame()


def seleccionar(tabla: pd.DataFrame, metrica: str = "deficit",
                percentiles=PERCENTILES) -> list[Escenario]:
    """Elige el año análogo más cercano a cada percentil de la métrica."""
    if tabla.empty:
        raise ValueError("No hay años con la ventana completa")
    if metrica not in METRICAS:
        raise KeyError(f"Métrica desconocida {metrica!r}; opciones {sorted(METRICAS)}")
    serie = tabla[metrica]
    nombres = NOMBRES[metrica]
    out = []
    for q in percentiles:
        valor = float(np.percentile(serie, q))
        anio = int((serie - valor).abs().idxmin())
        out.append(Escenario(nombres.get(q, f"P{q}"), q, metrica, valor, anio,
                             float(serie.loc[anio])))
    return out


def serie_escenario(clima: pd.DataFrame, anio_analogo: int, inicio: str | pd.Timestamp,
                    dias: int) -> pd.DataFrame:
    """Serie diaria del año análogo re-fechada a la campaña que empieza en ``inicio``.

    Se conserva el orden de los días (un desfase por año bisiesto se acepta y se documenta).
    """
    inicio = pd.Timestamp(inicio).normalize()
    try:
        a = pd.Timestamp(anio_analogo, inicio.month, inicio.day)
    except ValueError:
        a = pd.Timestamp(anio_analogo, inicio.month, 28)
    v = clima.loc[a:a + pd.Timedelta(days=dias - 1), ["tmax", "tmin", "et0", "precip"]].copy()
    if len(v) < dias:
        raise ValueError(f"El año análogo {anio_analogo} no cubre {dias} días desde {a.date()}")
    v.index = pd.date_range(inicio, periods=dias)
    v.attrs["anio_analogo"] = anio_analogo
    return v


def robustez(tabla: pd.DataFrame) -> str:
    """«normal» con ≥ 15 años; «limitada» con menos (EX-06)."""
    return "normal" if len(tabla) >= MIN_ANIOS_ROBUSTO else "limitada"
