"""Simulación de una campaña: Kc → ETc → balance FAO-56 → riego, cobertura y autonomía.

Cada cifra devuelta es un :class:`~nortex_motor.valores.Valor` con unidad, categoría y fuente.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nortex_motor import escenarios as esc
from nortex_motor.balance import TOLERANCIA_CIERRE_MM, balance_fao56, cierre
from nortex_motor.catalogo import Cultivo, cultivo
from nortex_motor.clima import validar
from nortex_motor.kc import curva_kc
from nortex_motor.riego import (
    Riego,
    autonomia_dias,
    cobertura_hidrica,
    dias_de_turno,
    dias_estres_por_turno,
    lamina_por_turno,
)
from nortex_motor.suelo import Suelo
from nortex_motor.termico import dias_sobre_umbral, gdd
from nortex_motor.valores import Valor

BALANCE = "balance FAO-56 (ec. 85–88)"


@dataclass
class ResultadoCampania:
    """Resultado de simular una campaña de un cultivo con un clima dado."""

    cultivo: str
    inicio: pd.Timestamp
    fin: pd.Timestamp
    escenario: esc.Escenario | None
    valores: dict[str, Valor]
    diario: pd.DataFrame
    advertencias: list[str] = field(default_factory=list)

    def tabla(self) -> pd.DataFrame:
        """Valores como tabla (una fila por indicador)."""
        return pd.DataFrame({k: v.a_dict() for k, v in self.valores.items()}).T


def _clima_campania(clima: pd.DataFrame, fechas: pd.DatetimeIndex) -> pd.DataFrame:
    validar(clima)
    v = clima.reindex(fechas)[["tmax", "tmin", "et0", "precip"]]
    faltan = v["et0"].isna().sum()
    if faltan:
        raise ValueError(f"El clima no cubre la campaña: {faltan} días sin ET0 entre "
                         f"{fechas[0].date()} y {fechas[-1].date()}")
    return v


def simular_campania(
    cultivo_id: str,
    inicio: str | pd.Timestamp,
    suelo: Suelo,
    riego: Riego,
    clima: pd.DataFrame,
    area_ha: float = 1.0,
    dr0: float = 0.0,
    escenario: esc.Escenario | None = None,
    dias: int | None = None,
) -> ResultadoCampania:
    """Simula una campaña completa.

    ``clima`` es una tabla diaria con ``tmax, tmin, et0, precip`` que cubre la campaña (una
    serie de escenario o un año observado). ``inicio`` es la fecha de siembra (anuales) o de
    inicio de campaña (perennes, 365 días por defecto).
    """
    c: Cultivo = cultivo(cultivo_id)
    kc = curva_kc(c, inicio, dias)
    fechas = kc.index
    cl = _clima_campania(clima, fechas)
    cat_clima = "escenario" if escenario is not None else "modelado"
    origen = escenario.etiqueta if escenario is not None else "clima observado (POWER)"
    adv: list[str] = []
    V: dict[str, Valor] = {}

    def sim(nombre, valor, unidad, fuente=BALANCE, nota=""):
        V[nombre] = Valor(valor, unidad, "simulado", f"{fuente}; {origen}", nota)

    V["clima"] = Valor(origen, "-", cat_clima, "NASA POWER diario")
    V["ciclo"] = Valor(len(fechas), "días", "declarado" if c.perenne else "simulado",
                       c.fuente("etapas_dias") if not c.perenne else "campaña anual")
    V["et0_total"] = Valor(float(cl["et0"].sum()), "mm", cat_clima, "POWER + ET0 FAO-56")
    V["precip_total"] = Valor(float(cl["precip"].sum()), "mm", cat_clima, "POWER PRECTOTCORR")

    etc = kc["kc"] * cl["et0"]
    demanda_neta = float((etc - cl["precip"]).clip(lower=0).sum())
    zr, p = float(c.get("zr_m")), float(c.get("p"))

    if c.anegado:
        perdidas = float(c.get("perdidas_mm_dia") or 0.0)
        demanda_neta = float((etc + perdidas - cl["precip"]).clip(lower=0).sum())
        diario = pd.DataFrame({"et0": cl["et0"], "kc": kc["kc"], "etc": etc,
                               "precip": cl["precip"], "etapa": kc["etapa"]})
        adv.append("Cultivo anegado: el balance de suelo FAO-56 no aplica; la demanda incluye "
                   f"ETc y {perdidas} mm/día de pérdidas por percolación ({c.fuente('perdidas_mm_dia')}).")
        sim("etc_total", float(etc.sum()), "mm", "Kc FAO-56 × ET0")
    else:
        taw, raw = suelo.taw_mm(zr), suelo.raw_mm(zr, p)
        turnos = dias_de_turno(fechas, riego)
        if not turnos.any():
            adv.append("Sin turnos de riego declarados: la campaña se simula en secano.")
        if riego.modo == "dotacion" and riego.dotacion_m3_ha is not None:
            diario = balance_fao56(cl["et0"], cl["precip"], kc["kc"], taw, p,
                                   riego_neto=lamina_por_turno(fechas, riego), dr0=dr0)
            modo = "dotación repartida en turnos iguales"
        else:
            diario = balance_fao56(cl["et0"], cl["precip"], kc["kc"], taw, p,
                                   turnos_llenado=turnos, dr0=dr0)
            modo = "reposición a capacidad de campo en cada turno"
            if riego.dotacion_m3_ha is None:
                adv.append("Dotación no confirmada (EX-04): se simula reponiendo el suelo en cada "
                           "turno; la cobertura hídrica queda por determinar.")
        diario["etapa"] = kc["etapa"]
        err = cierre(diario)
        if abs(err) > TOLERANCIA_CIERRE_MM:
            raise RuntimeError(f"El balance no cierra: error {err:.3f} mm")

        V["taw"] = Valor(taw, "mm", "simulado", f"θ_FC−θ_WP ({suelo.fuente}) × Zr ({c.fuente('zr_m')})")
        V["raw"] = Valor(raw, "mm", "simulado", f"p ({c.fuente('p')}) × TAW")
        sim("etc_total", float(diario["etc"].sum()), "mm", "Kc FAO-56 × ET0")
        sim("eta_total", float(diario["etc_adj"].sum()), "mm")
        deficit = float(diario["etc"].sum() - diario["etc_adj"].sum())
        sim("deficit_transpiracion", deficit, "mm", nota="ETc − ETc,adj: agua que el cultivo no pudo usar")
        sim("dias_estres", int((diario["ks"] < 1).sum()), "días", nota="días con Ks < 1")
        critica = diario["etapa"].isin(["media", "perenne"])
        sim("dias_estres_etapa_critica", int((diario.loc[critica, "ks"] < 1).sum()), "días",
            nota="etapa media (floración y llenado) o todo el año en perennes")
        sim("ks_minimo", float(diario["ks"].min()), "-")
        sim("riego_neto", float(diario["riego"].sum()), "mm", nota=modo)
        sim("riego_bruto", float(diario["riego"].sum() / riego.eficiencia), "mm",
            nota=f"eficiencia {riego.eficiencia:.2f} (declarada)")
        sim("percolacion", float(diario["dp"].sum()), "mm")
        sim("cierre_balance", round(err, 4), "mm", nota="debe ser ≈ 0")

        etc_pico = float(diario.loc[critica, "etc"].mean()) if critica.any() else float(diario["etc"].mean())
        if c.perenne:  # mes de mayor demanda
            etc_pico = float(diario["etc"].resample("MS").mean().max())
        aut = autonomia_dias(raw, etc_pico)
        sim("autonomia_entre_turnos", aut, "días", "RAW / ETc en el periodo de mayor demanda")
        sim("estres_por_turno", dias_estres_por_turno(riego.intervalo_dias, aut), "días",
            "máx(0, intervalo − autonomía)")

    sim("demanda_neta", demanda_neta, "mm", "Σ máx(0, ETc − P)")
    cob = cobertura_hidrica(demanda_neta, riego.eficiencia, area_ha, riego.dotacion_m3_ha)
    V["cobertura_hidrica"] = Valor(None if cob["c"] is None else round(cob["c"], 3), "-",
                                   "simulado", f"dotación declarada / demanda bruta; {origen}",
                                   cob["etiqueta"])
    V["demanda_bruta"] = Valor(cob["demanda_bruta_m3"], "m³", "simulado",
                               f"demanda neta / eficiencia × área ({area_ha} ha)")

    if not c.perenne:
        g = gdd(cl["tmax"], cl["tmin"], float(c.get("tbase_c")), c.get("tupper_c"))
        V["gdd"] = Valor(float(g.sum()), "°C·día", "simulado",
                         f"Tbase {c.get('tbase_c')} °C ({c.fuente('tbase_c')}); {origen}")
        umbral = c.get("calor_floracion_c")
        if umbral is not None:
            media = kc["etapa"].eq("media")
            V["dias_calor_etapa_media"] = Valor(
                dias_sobre_umbral(cl["tmax"], float(umbral), media), "días", "simulado",
                f"Tmax > {umbral} °C ({c.fuente('calor_floracion_c')}); {origen}")

    a, b = c.get("ece_umbral"), c.get("ece_pendiente")
    if suelo.ece is None:
        V["rendimiento_relativo_salinidad"] = Valor(None, "%", "simulado", "sin análisis de suelo",
                                                    "por determinar (EX-03)")
    elif b is None:
        nota = "ECe supera el umbral del cultivo" if a is not None and suelo.ece > a else "ECe bajo el umbral"
        V["rendimiento_relativo_salinidad"] = Valor(None, "%", "simulado",
                                                    f"umbral {a} dS/m ({c.fuente('ece_umbral')})",
                                                    f"{nota}; pendiente sin dato")
    else:
        yr = max(0.0, 100.0 - b * max(0.0, suelo.ece - a))
        V["rendimiento_relativo_salinidad"] = Valor(yr, "%", "simulado",
                                                    f"Maas–Hoffman a={a}, b={b} ({c.fuente('ece_umbral')})")

    if any(p.estado == "POR_VALIDAR" for p in c.parametros.values()):
        adv.append("Parámetros del cultivo POR VALIDAR (rangos de literatura); usar como orientación.")
    return ResultadoCampania(cultivo_id, fechas[0], fechas[-1], escenario, V, diario, adv)


def simular_escenarios(
    cultivo_id: str,
    inicio: str | pd.Timestamp,
    suelo: Suelo,
    riego: Riego,
    clima_historico: pd.DataFrame,
    area_ha: float = 1.0,
    metrica: str = "deficit",
    anios: list[int] | None = None,
    dias: int | None = None,
) -> dict:
    """Simula la campaña con los escenarios P10/P50/P90 construidos desde el histórico.

    Devuelve ``{"escenarios": [...], "resultados": {nombre: ResultadoCampania},
    "ventanas": DataFrame, "robustez": "normal" | "limitada"}``.
    """
    c = cultivo(cultivo_id)
    n = dias or c.ciclo_dias or 365
    inicio = pd.Timestamp(inicio)
    ventanas = esc.ventanas_historicas(clima_historico, inicio.strftime("%m-%d"), n, anios)
    escenarios = esc.seleccionar(ventanas, metrica)
    resultados = {}
    for e in escenarios:
        serie = esc.serie_escenario(clima_historico, e.anio_analogo, inicio, n)
        resultados[e.nombre] = simular_campania(cultivo_id, inicio, suelo, riego, serie,
                                                area_ha, escenario=e, dias=dias)
    return {"escenarios": escenarios, "resultados": resultados, "ventanas": ventanas,
            "robustez": esc.robustez(ventanas)}


def resumen(resultados: dict[str, ResultadoCampania], indicadores=None) -> pd.DataFrame:
    """Tabla comparativa (una columna por escenario) con los valores numéricos."""
    indicadores = indicadores or [
        "et0_total", "etc_total", "eta_total", "deficit_transpiracion", "dias_estres",
        "dias_estres_etapa_critica", "riego_bruto", "demanda_neta", "cobertura_hidrica",
        "autonomia_entre_turnos", "estres_por_turno", "dias_calor_etapa_media",
    ]
    data = {}
    for nombre, r in resultados.items():
        data[nombre] = {k: r.valores[k].valor for k in indicadores if k in r.valores}
    return pd.DataFrame(data)
