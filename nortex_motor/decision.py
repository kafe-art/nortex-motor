"""Motor de evaluación: cuatro dimensiones separadas y funciones F1 y F2 (v0.7 §3, §4, §14).

Regla de no invención (v0.7 §2): **no hay índice compuesto**. Cada dimensión se reporta por
separado con su evidencia, su fuente y los umbrales usados (declarados abajo, POR VALIDAR):

- **Viabilidad**: «viable», «viable con restricciones», «por determinar» o «no viable», con
  los motivos (cobertura hídrica RT-06, área autorizada RT-08, salinidad, compromiso perenne).
- **Riesgo**: lista de factores con nivel bajo / medio / alto y la cifra que lo sustenta.
- **Conveniencia**: margen bruto de referencia ajustado por estrés hídrico (Ky, FAO-33) y
  salinidad, y margen por m³ de agua.
- **Fundamentación**: alta / media / limitada según la procedencia de los insumos, con la
  lista de datos que faltan (RT-07).

F1 compara ventanas de siembra de un cultivo; F2 compara cultivos para la misma ventana y
clasifica la transición desde el historial (continuidad, cambio, transición, renovación).
Para ordenar se usa una regla lexicográfica declarada, no un puntaje.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from nortex_motor.catalogo import cultivo as get_cultivo
from nortex_motor.pipeline import Contexto, Evaluacion, _ultimo, evaluar_ctx
from nortex_motor.termico import tmin_media_invierno

#: Umbrales de riesgo y viabilidad (POR VALIDAR con técnicos locales).
UMBRALES = {
    "estres_critico_dias": (1, 8),      # días con Ks < 1 en etapa crítica: < 1 bajo, < 8 medio, ≥ 8 alto
    "estres_por_turno_dias": (1, 4),    # días de estrés por turno
    "calor_floracion_dias": (1, 4),     # días con Tmax > umbral del cultivo en etapa media
    "icen_moderado": 1.0,               # ENFEN: El Niño Costero moderado (POR VERIFICAR)
    "icen_fuerte": 1.7,                 # ENFEN: El Niño Costero fuerte (POR VERIFICAR)
    "anomalia_tmin_mango": 1.0,         # °C sobre la mediana del invierno en años El Niño
    "rend_salino_no_viable": 50.0,      # % de rendimiento relativo por salinidad
    "rend_salino_restriccion": 90.0,
}
NIVELES = ("bajo", "medio", "alto")
ORDEN_VIABILIDAD = {"viable": 0, "viable con restricciones": 1, "por determinar": 2, "no viable": 3}
SENSIBLES_ANEGAMIENTO = {"caupi", "maiz", "limon_sutil", "mango"}


def _nivel(valor, cortes) -> str:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "por determinar"
    bajo, alto = cortes
    return "bajo" if valor < bajo else "medio" if valor < alto else "alto"


def _v(ev: Evaluacion, escenario: str, nombre: str):
    r = ev.resultados.get(escenario)
    return None if r is None or nombre not in r.valores else r.valores[nombre].valor


@dataclass
class Decision:
    """Evaluación de un cultivo en una ventana, en cuatro dimensiones separadas."""

    cultivo: str
    inicio: pd.Timestamp
    viabilidad: dict
    riesgo: list[dict]
    conveniencia: dict
    fundamentacion: dict
    evaluacion: Evaluacion = field(repr=False)

    def a_dict(self) -> dict:
        """Representación serializable para la API web."""
        ev = self.evaluacion
        escenarios = {
            nombre: {k: v.a_dict() for k, v in r.valores.items()}
            for nombre, r in ev.resultados.items()
        }
        return {
            "cultivo": self.cultivo,
            "inicio": self.inicio.date().isoformat(),
            "viabilidad": self.viabilidad,
            "riesgo": self.riesgo,
            "conveniencia": self.conveniencia,
            "fundamentacion": self.fundamentacion,
            "escenarios": [asdict(e) for e in ev.escenarios],
            "resultados": escenarios,
            "hindcast": {k: v for k, v in ev.hindcast.items() if k != "detalle"},
            "procedencia": ev.procedencia,
            "advertencias": ev.advertencias,
        }


# --- Dimensiones ------------------------------------------------------------------------

def viabilidad(ev: Evaluacion, ctx: Contexto) -> dict:
    """Viabilidad con motivos; manda el peor estado encontrado."""
    c = get_cultivo(ev.cultivo)
    motivos: list[tuple[str, str]] = []

    area = _ultimo(ctx.local, "pcr_area_authorized", f"cultivo={ev.cultivo}")
    if area is not None and area[0] <= 0:
        motivos.append(("no viable", "El PCR no autoriza área para este cultivo (RT-08)."))

    c_medio, c_desf = _v(ev, "medio", "cobertura_hidrica"), _v(ev, "desfavorable", "cobertura_hidrica")
    if c_medio is None:
        motivos.append(("por determinar", "Cobertura hídrica sin dotación confirmada (EX-04)."))
    elif c_medio < 1:
        motivos.append(("no viable", f"La dotación cubre el {c_medio:.0%} de la demanda en el escenario "
                                     "medio (RT-06)."))
    elif c_desf is not None and c_desf < 1:
        motivos.append(("viable con restricciones",
                        f"Cubre la demanda en el escenario medio, pero solo el {c_desf:.0%} en el "
                        "desfavorable."))

    sal = ev.resultados["medio"].valores.get("rendimiento_relativo_salinidad")
    if sal is not None and sal.valor is not None:
        if sal.valor < UMBRALES["rend_salino_no_viable"]:
            motivos.append(("no viable",
                            f"Salinidad: rendimiento relativo {sal.valor:.0f} % (Maas–Hoffman)."))
        elif sal.valor < UMBRALES["rend_salino_restriccion"]:
            motivos.append(("viable con restricciones",
                            f"Salinidad: rendimiento relativo {sal.valor:.0f} % (Maas–Hoffman)."))
    elif sal is not None and "supera" in sal.nota:
        motivos.append(("viable con restricciones", "La ECe supera el umbral del cultivo (sin pendiente "
                                                    "para cuantificar la pérdida)."))

    if c.perenne:
        juvenil = c.get("periodo_juvenil_anios") or [3, 4]
        vida = c.get("vida_util_anios") or [20, 30]
        motivos.append(("viable con restricciones",
                        f"Compromiso de {vida[0]}–{vida[1]} años y {juvenil[0]}–{juvenil[1]} años sin "
                        "ingreso (RT-02, RF-21)."))

    estado = max((m[0] for m in motivos), key=ORDEN_VIABILIDAD.get, default="viable")
    return {"estado": estado, "motivos": [m[1] for m in motivos] or ["Sin incompatibilidades detectadas."]}


def riesgo(ev: Evaluacion, ctx: Contexto) -> list[dict]:
    """Factores de riesgo con nivel, evidencia y fuente."""
    c = get_cultivo(ev.cultivo)
    items: list[dict] = []

    def add(factor, nivel, evidencia, fuente):
        items.append({"factor": factor, "nivel": nivel, "evidencia": evidencia, "fuente": fuente})

    if not c.anegado:
        criticos = [_v(ev, e, "dias_estres_etapa_critica") for e in ("medio", "desfavorable")]
        if all(x is not None for x in criticos):
            add("Estrés hídrico en etapa crítica", _nivel(max(criticos), UMBRALES["estres_critico_dias"]),
                f"{criticos[0]} días con Ks < 1 (medio) y {criticos[1]} (desfavorable)",
                "balance FAO-56 (simulado)")
        turno = _v(ev, "medio", "estres_por_turno")
        aut = _v(ev, "medio", "autonomia_entre_turnos")
        if turno is not None:
            add("Turno de riego más largo que la autonomía del suelo",
                _nivel(turno, UMBRALES["estres_por_turno_dias"]),
                f"el suelo aguanta {aut:.1f} días; {turno:.1f} días de estrés por turno",
                "RAW / ETc (simulado) + turnos declarados")

    calor = [_v(ev, e, "dias_calor_etapa_media") for e in ("medio", "desfavorable")]
    if all(x is not None for x in calor):
        add("Calor en floración", _nivel(max(calor), UMBRALES["calor_floracion_dias"]),
            f"hasta {max(calor)} días con Tmax sobre el umbral del cultivo", "clima por escenario")

    enso = ev.enso or {}
    if enso.get("icen_ultimo"):
        mes, icen = enso["icen_ultimo"]
        fechas = ev.resultados["medio"].diario.index
        if c.id in SENSIBLES_ANEGAMIENTO and fechas.month.isin([1, 2, 3, 4]).any():
            nivel = ("alto" if icen >= UMBRALES["icen_fuerte"] else
                     "medio" if icen >= UMBRALES["icen_moderado"] else "bajo")
            evid = f"ICEN {mes} = {icen:+.2f}"
            nino = enso.get("anios_nino") or []
            if nino:
                clima = ctx.clima
                ventana = [clima.loc[pd.Timestamp(a, fechas[0].month, min(fechas[0].day, 28)):
                                     pd.Timestamp(a, fechas[0].month, min(fechas[0].day, 28))
                                     + pd.Timedelta(days=len(fechas) - 1), "precip"].sum()
                           for a in nino]
                evid += (f"; en años El Niño análogos la lluvia de la campaña llegó a "
                         f"{max(ventana):.0f} mm")
            add("Lluvias intensas y encharcamiento (El Niño Costero)", nivel, evid,
                "ICEN observado; persistencia por confirmar con el comunicado ENFEN (pronosticado)")
        if c.id == "mango":
            anom = tmin_media_invierno(ctx.clima["tmin"])
            nino = [a for a in enso.get("anios_nino", []) if a in anom.index]
            if len(anom) and nino:
                exceso = float(anom.loc[nino].mean() - anom.median())
                nivel = ("alto" if icen >= UMBRALES["icen_moderado"]
                         and exceso >= UMBRALES["anomalia_tmin_mango"] else
                         "medio" if icen >= 0.4 else "bajo")
                add("Falla de inducción floral por invierno cálido", nivel,
                    f"ICEN {mes} = {icen:+.2f}; en años El Niño la Tmin de jun–ago fue "
                    f"{exceso:+.1f} °C sobre la mediana", "POWER (corregido) + ICEN")

    est = ctx.estado_reservorio
    if est and est.get("categoria") == "bajo":
        add("Reservorio bajo para la época", "alto",
            f"{est['volumen_hm3']:.0f} hm³ ({est['porcentaje_capacidad']} %), percentil {est['percentil']}",
            "volumen observado")
    return items


def conveniencia(ev: Evaluacion, ctx: Contexto, economia: dict | None = None) -> dict:
    """Margen bruto de referencia ajustado por agua (Ky) y salinidad, por ha y por m³."""
    c = get_cultivo(ev.cultivo)
    datos = (economia or {}).get(ev.cultivo)
    fuente = "parámetro"
    if datos is None:
        r = _ultimo(ctx.local, "yield_reference", f"cultivo={ev.cultivo}")
        p = _ultimo(ctx.local, "price_farmgate", f"cultivo={ev.cultivo}")
        k = _ultimo(ctx.local, "cost_production", f"cultivo={ev.cultivo}")
        if r and p and k:
            datos = {"rendimiento_t_ha": r[0], "precio_soles_kg": p[0], "costo_soles_ha": k[0]}
            fuente = f"declarado: {r[1]}"
    out: dict = {"fuente": fuente if datos else None}
    agua_m3 = _v(ev, "medio", "riego_bruto")
    agua_m3 = agua_m3 * 10 if agua_m3 else _v(ev, "medio", "demanda_bruta")
    out["agua_m3_ha"] = None if agua_m3 is None else round(agua_m3)
    if c.perenne:
        out["nota"] = "Perenne: durante el periodo juvenil el margen de la campaña es ≤ 0 (RT-02)."
    if not datos:
        out.update(estado="por determinar", nota=out.get("nota", "") +
                   " Sin rendimiento, precio y costo de referencia (SIEA/MIDAGRI).")
        return out

    ky = c.get("ky")
    factores = {}
    for esc in ("favorable", "medio", "desfavorable"):
        etc, eta = _v(ev, esc, "etc_total"), _v(ev, esc, "eta_total")
        f_agua = max(0.0, 1 - ky * (1 - eta / etc)) if ky and etc and eta is not None else 1.0
        sal = _v(ev, esc, "rendimiento_relativo_salinidad")
        f_sal = sal / 100 if sal is not None else 1.0
        factores[esc] = min(f_agua, f_sal)
    margen = {}
    for esc, f in factores.items():
        rend = datos["rendimiento_t_ha"] * f
        margen[esc] = rend * 1000 * datos["precio_soles_kg"] - datos["costo_soles_ha"]
    out.update(
        estado="calculado",
        rendimiento_referencia_t_ha=datos["rendimiento_t_ha"],
        rendimiento_relativo={k: round(v, 3) for k, v in factores.items()},
        margen_soles_ha={k: round(v) for k, v in margen.items()},
        margen_soles_m3=(None if not agua_m3 else round(margen["medio"] / agua_m3, 2)),
        metodo=("rend. ref. × mín(1 − Ky·(1 − ETa/ETc), rend. relativo por salinidad) × precio − costo"
                + ("" if ky else "; sin Ky: no se ajusta por estrés hídrico")),
    )
    return out


def fundamentacion(ev: Evaluacion, conv: dict) -> dict:
    """Nivel de fundamentación según la procedencia de cada insumo y datos faltantes (RT-07)."""
    p = ev.procedencia
    insumos = {
        "clima": "buena" if "corregido" in p.get("clima", "") else "media",
        "suelo": "buena" if p.get("suelo", "").startswith("laboratorio") else
                 "media" if "SoilGrids" in p.get("suelo", "") else "débil",
        "dotacion": "débil" if p.get("dotacion", "").startswith("por determinar") else
                    "media" if p.get("dotacion", "").startswith("escenario") else "buena",
        "turnos": "débil" if p.get("turnos", "").startswith("por determinar") else "buena",
        "parametros_cultivo": "media",  # todos POR VALIDAR hasta tener fichas INIA
        "escenarios": "buena" if ev.hindcast.get("veredicto", "").startswith("calibrado") else "débil",
        "economia": "buena" if conv.get("estado") == "calculado" else "débil",
    }
    esenciales = ("suelo", "dotacion", "turnos")
    if any(insumos[k] == "débil" for k in esenciales):
        nivel = "limitada"
    elif any(v != "buena" for v in insumos.values()):
        nivel = "media"
    else:
        nivel = "alta"
    faltantes = {
        "suelo": "análisis de suelo (θ, ECe) de la parcela",
        "dotacion": "dotación del PCR de la campaña (Junta de Usuarios)",
        "turnos": "intervalo de turnos de la comisión",
        "clima": "estación de referencia en el valle (SENAMHI)",
        "parametros_cultivo": "fichas INIA del cultivo (Kc, umbrales)",
        "escenarios": "más años de historia o revisar la métrica de escenarios",
        "economia": "rendimiento, precio y costo de Piura (SIEA/MIDAGRI)",
    }
    return {"nivel": nivel, "insumos": insumos,
            "faltantes": [faltantes[k] for k, v in insumos.items() if v != "buena"]}


def decidir(ctx: Contexto, cultivo_id: str, inicio, economia: dict | None = None, **kw) -> Decision:
    """Evalúa un cultivo en una ventana y devuelve las cuatro dimensiones."""
    ev = evaluar_ctx(ctx, cultivo_id, inicio, **kw)
    conv = conveniencia(ev, ctx, economia)
    return Decision(cultivo_id, ev.inicio, viabilidad(ev, ctx), riesgo(ev, ctx), conv,
                    fundamentacion(ev, conv), ev)


# --- F1 y F2 ------------------------------------------------------------------------------

def _fila(d: Decision) -> dict:
    niveles = [r["nivel"] for r in d.riesgo]
    ev = d.evaluacion
    return {
        "cultivo": d.cultivo,
        "inicio": d.inicio.date(),
        "viabilidad": d.viabilidad["estado"],
        "riesgos_altos": niveles.count("alto"),
        "riesgos_medios": niveles.count("medio"),
        "estres_critico_medio": _v(ev, "medio", "dias_estres_etapa_critica"),
        "cobertura_medio": _v(ev, "medio", "cobertura_hidrica"),
        "cobertura_desfavorable": _v(ev, "desfavorable", "cobertura_hidrica"),
        "demanda_neta_mm": _v(ev, "medio", "demanda_neta"),
        "margen_medio_soles_ha": (d.conveniencia.get("margen_soles_ha") or {}).get("medio"),
        "fundamentacion": d.fundamentacion["nivel"],
    }


def _ordenar(tabla: pd.DataFrame) -> pd.DataFrame:
    """Orden lexicográfico declarado: viabilidad → riesgos altos → riesgos medios → demanda."""
    return (tabla.assign(_v=tabla["viabilidad"].map(ORDEN_VIABILIDAD))
            .sort_values(["_v", "riesgos_altos", "riesgos_medios", "demanda_neta_mm"])
            .drop(columns="_v").reset_index(drop=True))


def f1_ventanas(ctx: Contexto, cultivo_id: str, desde, hasta, paso_dias: int = 15,
                economia: dict | None = None, **kw) -> dict:
    """F1 — ¿cuándo conviene sembrar? Evalúa ventanas cada ``paso_dias`` entre dos fechas."""
    fechas = pd.date_range(desde, hasta, freq=f"{paso_dias}D")
    decisiones = [decidir(ctx, cultivo_id, f, economia, **kw) for f in fechas]
    tabla = _ordenar(pd.DataFrame([_fila(d) for d in decisiones]))
    return {"tabla": tabla, "decisiones": decisiones,
            "regla_orden": "viabilidad → menos riesgos altos → menos riesgos medios → menor demanda"}


def tipo_transicion(previo: str | None, objetivo: str) -> str:
    """Continuidad, cambio, transición (a perenne) o renovación (desde perenne)."""
    if previo is None:
        return "sin historial"
    if previo == objetivo:
        return "continuidad"
    if get_cultivo(previo).perenne:
        return "renovación"
    return "transición" if get_cultivo(objetivo).perenne else "cambio"


def reglas_transicion(ctx: Contexto, historial: list[str], objetivo: str) -> list[str]:
    """Reglas RT-01…RT-07 aplicables a la transición desde el historial."""
    previo = historial[-1] if historial else None
    out = []
    if previo and get_cultivo(previo).perenne and previo != objetivo:
        out.append("RT-01: el perenne en producción solo cambia con una decisión explícita de renovación.")
    if get_cultivo(objetivo).perenne and previo != objetivo:
        out.append("RT-02: instalar el perenne implica 3–4 años sin ingreso y un compromiso de ~20 años.")
    if objetivo == "arroz" and historial[-2:] == ["arroz", "arroz"]:
        out.append("RT-04 (informativa): dos campañas seguidas de arroz; conviene revisarlo con un técnico.")
    if previo == "arroz" and get_cultivo(objetivo).perenne:
        faltan = [n for v, n in (("water_table_depth", "napa"), ("soil_ec", "ECe del suelo"),
                                 ("water_ec", "ECw del agua")) if _ultimo(ctx.local, v) is None]
        out.append("RT-05: verificar drenaje y salinidad antes de pasar de arroz a frutal"
                   + (f"; faltan: {', '.join(faltan)}" if faltan else "; datos disponibles."))
    return out


def f2_cultivos(ctx: Contexto, cultivos: list[str], inicio, historial: list[str] | None = None,
                economia: dict | None = None, **kw) -> dict:
    """F2 — ¿qué conviene sembrar? Compara cultivos en una ventana y la transición desde el historial."""
    historial = list(historial or [])
    decisiones = [decidir(ctx, c, inicio, economia, **kw) for c in cultivos]
    filas = []
    for d in decisiones:
        fila = _fila(d)
        fila["transicion"] = tipo_transicion(historial[-1] if historial else None, d.cultivo)
        reglas = reglas_transicion(ctx, historial, d.cultivo)
        fila["reglas"] = " | ".join(r.split(":")[0] for r in reglas) or "—"
        filas.append(fila)
        d.viabilidad.setdefault("reglas_transicion", reglas)
    return {"tabla": _ordenar(pd.DataFrame(filas)), "decisiones": decisiones,
            "regla_orden": "viabilidad → menos riesgos altos → menos riesgos medios → menor demanda"}
