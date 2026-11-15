"""Evaluación de punta a punta: alimenta el motor con todas las fuentes de `nortex_data`.

Para cada insumo se elige la mejor fuente disponible y se registra su procedencia:

| Insumo | Prioridad |
| :-- | :-- |
| Clima | POWER corregido con estación (SENAMHI/NOAA) → POWER sin corregir |
| Suelo | Laboratorio (plantilla) → SoilGrids → textura por defecto (EX-03) |
| Dotación | Parámetro → PCR declarado → dotación análoga por volumen (EX-15) → por determinar |
| Turnos | Parámetro → plantilla de turnos → por determinar (EX-16) |
| Eficiencia | Parámetro → plantilla de riego → 0,5 gravedad (POR VALIDAR) |
| Escenarios | Años filtrados por ENSO (ICEN) si se pide y hay ≥ 10 años → todos los años (EX-07) |

Las entradas son DataFrames en el formato largo de `nortex_data`
(``date, aoi_id, lat, lon, source, product, variable, value, unit, qc_flag``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nortex_motor import reservorio as res
from nortex_motor.catalogo import cultivo as get_cultivo
from nortex_motor.clima import desde_largo
from nortex_motor.hindcast import hindcast
from nortex_motor.riego import EFICIENCIA_REFERENCIA, Riego
from nortex_motor.simulacion import resumen, simular_escenarios
from nortex_motor.suelo import TEXTURAS_FAO56, Suelo
from nortex_motor.validacion import estacion_desde_largo, validar_y_calibrar

#: Umbrales del ICEN medio de la ventana para clasificar años (POR VERIFICAR con la nota
#: técnica de ENFEN 2024; El Niño Costero > +0,4; La Niña Costera < −1,0).
UMBRAL_NINO, UMBRAL_NINA = 0.4, -1.0
MAX_ANTIGUEDAD_RESERVORIO = 30  # días
CAPAS_SOILGRIDS_CM = {"0-5cm": (0, 5), "5-15cm": (5, 15), "15-30cm": (15, 30),
                      "30-60cm": (30, 60), "60-100cm": (60, 100)}


@dataclass
class Evaluacion:
    """Resultado de :func:`evaluar`."""

    cultivo: str
    inicio: pd.Timestamp
    suelo: Suelo
    riego: Riego
    resultados: dict
    escenarios: list
    resumen: pd.DataFrame
    validacion: dict | None
    hindcast: dict
    reservorio: dict | None
    enso: dict | None
    procedencia: dict
    advertencias: list[str] = field(default_factory=list)


def _ultimo(df: pd.DataFrame | None, variable: str, contiene: str | None = None):
    if df is None or df.empty:
        return None
    sel = df[(df["variable"] == variable) & df["value"].notna()]
    if contiene:
        sel = sel[sel["product"].str.lower().str.contains(contiene.lower(), na=False)]
    if sel.empty:
        return None
    fila = sel.sort_values("date").iloc[-1]
    return float(fila["value"]), str(fila["product"])


def _suelo(c, local, suelo_datos, adv, proc) -> Suelo:
    fc, wp = _ultimo(local, "lab_theta_fc"), _ultimo(local, "lab_theta_wp")
    ece = _ultimo(local, "soil_ec")
    ece_v = ece[0] if ece else None
    if fc and wp and fc[0] > wp[0]:
        proc["suelo"] = f"laboratorio (declarado): {fc[1]}"
        return Suelo(fc[0], wp[0], ece=ece_v, fuente="laboratorio")
    if suelo_datos is not None and not suelo_datos.empty:
        zr_cm = 100 * float(c.get("zr_m"))
        pesos = {}
        for capa, (a, b) in CAPAS_SOILGRIDS_CM.items():
            espesor = max(0.0, min(b, zr_cm) - a)
            if espesor > 0:
                pesos[capa] = espesor
        def media(var):
            d = suelo_datos[(suelo_datos["variable"] == var) & suelo_datos["value"].notna()]
            d = d.assign(capa=d["product"].str.split(" / ").str[-1])
            d = d[d["capa"].isin(pesos)]
            if d.empty:
                return None
            w = d["capa"].map(pesos)
            return float((d["value"] * w).sum() / w.sum())
        sfc, swp = media("soil_theta_fc"), media("soil_theta_wp")
        if sfc and swp and sfc > swp:
            proc["suelo"] = f"SoilGrids 2.0 (global, 250 m), media ponderada hasta Zr = {zr_cm:.0f} cm"
            adv.append("Suelo de SoilGrids (dato global, EX-03): confirmar con análisis de laboratorio.")
            return Suelo(sfc, swp, ece=ece_v, fuente="SoilGrids 2.0")
    adv.append("Sin datos de suelo: se usa textura franca de FAO-56 (EX-03).")
    proc["suelo"] = "FAO-56 tabla 19, franco (por defecto)"
    base = TEXTURAS_FAO56["franco"]
    return Suelo(base.theta_fc, base.theta_wp, ece=ece_v, fuente=base.fuente)


def _clasificar_enso(enso, inicio, dias, anios):
    icen = enso[enso["variable"] == "icen"].set_index("date")["value"].sort_index()
    clases = {}
    for anio in anios:
        a = pd.Timestamp(anio, inicio.month, min(inicio.day, 28))
        meses = icen.loc[a.replace(day=1):a + pd.Timedelta(days=dias)]
        if len(meses):
            m = float(meses.mean())
            clases[anio] = "nino" if m > UMBRAL_NINO else "nina" if m < UMBRAL_NINA else "neutro"
    ultimo = (icen.index[-1].strftime("%Y-%m"), float(icen.iloc[-1])) if len(icen) else None
    return clases, ultimo


@dataclass
class Contexto:
    """Datos preparados una sola vez para evaluar varios cultivos o ventanas (F1/F2)."""

    clima: pd.DataFrame
    validacion: dict | None
    local: pd.DataFrame | None
    suelo_datos: pd.DataFrame | None
    enso: pd.DataFrame | None
    reservorios: pd.DataFrame | None
    estado_reservorio: dict | None
    fecha_consulta: pd.Timestamp
    procedencia: dict
    advertencias: list[str]
    lluvia: pd.Series | None = None      # lluvia diaria de IMERG (para extremos)
    relieve: pd.DataFrame | None = None  # SRTM: elevation, slope, relief


def preparar(
    clima: pd.DataFrame,
    *,
    estacion: pd.DataFrame | None = None,
    nombre_estacion: str = "estación",
    suelo_datos: pd.DataFrame | None = None,
    local: pd.DataFrame | None = None,
    enso: pd.DataFrame | None = None,
    reservorios: pd.DataFrame | None = None,
    fecha_consulta: str | pd.Timestamp | None = None,
    lluvia_datos: pd.DataFrame | None = None,
    relieve: pd.DataFrame | None = None,
) -> Contexto:
    """Carga el clima, lo valida y corrige con la estación y resume el reservorio.

    ``lluvia_datos`` (IMERG, formato largo) se usa para los extremos de lluvia; ``relieve``
    (SRTM) para el factor de drenaje del riesgo por exceso de lluvia.
    """
    fecha_consulta = pd.Timestamp(fecha_consulta or pd.Timestamp.today()).normalize()
    adv: list[str] = []
    proc: dict[str, str] = {}
    tabla = desde_largo(clima) if "variable" in clima.columns else clima.copy()
    proc["clima"] = "NASA POWER diario (modelado, celda ~50 km)"
    validacion = None
    if estacion is not None and not estacion.empty:
        tabla, validacion = validar_y_calibrar(tabla, estacion_desde_largo(estacion), nombre_estacion)
        adv.extend(validacion["advertencias"])
        if validacion["corregido"]:
            proc["clima"] += f", corregido con {nombre_estacion} (sesgo mensual; ET0 por Hargreaves)"
    else:
        adv.append("Sin estación de referencia: el clima no se valida ni se corrige (§19).")

    estado_res = None
    if reservorios is not None and not reservorios.empty:
        serie = res.serie_volumen(reservorios)
        if len(serie):
            estado_res = res.estado(serie, fecha_consulta)
            proc["reservorio"] = "volumen observado (ANA / Junta / reportes SENAMHI)"
            dias = estado_res.get("antiguedad_dias")
            if dias is not None and dias > MAX_ANTIGUEDAD_RESERVORIO:
                adv.append(f"El último volumen del reservorio es de {estado_res['fecha_dato']} "
                           f"({dias} días): no describe la situación actual (RNF-03).")
    lluvia = None
    if lluvia_datos is not None and not lluvia_datos.empty:
        sel = lluvia_datos[lluvia_datos["variable"].isin(["precipitation", "precip_daily"])]
        if len(sel):
            lluvia = sel.set_index(pd.to_datetime(sel["date"]))["value"].sort_index()
            lluvia = lluvia[~lluvia.index.duplicated(keep="last")]
            proc["lluvia_extremos"] = "GPM IMERG (0,1°)"
    return Contexto(tabla, validacion, local, suelo_datos, enso, reservorios, estado_res,
                    fecha_consulta, proc, adv, lluvia, relieve)


def evaluar_ctx(
    ctx: Contexto,
    cultivo_id: str,
    inicio: str | pd.Timestamp,
    *,
    area_ha: float = 1.0,
    intervalo_dias: int | None = None,
    dotacion_m3_ha: float | None = None,
    eficiencia: float | None = None,
    condicion_enso: str | None = None,
    metrica: str = "deficit",
) -> Evaluacion:
    """Evalúa un cultivo y una fecha de siembra sobre un contexto ya preparado."""
    c = get_cultivo(cultivo_id)
    inicio = pd.Timestamp(inicio).normalize()
    adv: list[str] = list(ctx.advertencias)
    proc: dict[str, str] = dict(ctx.procedencia)
    local, tabla, estado_res, reservorios = ctx.local, ctx.clima, ctx.estado_reservorio, ctx.reservorios

    suelo = _suelo(c, local, ctx.suelo_datos, adv, proc)

    origen_dot = "parámetro"
    if dotacion_m3_ha is None:
        hallado = _ultimo(local, "water_allocation", f"cultivo={cultivo_id}") or \
            _ultimo(local, "water_allocation", f"cultivo={c.nombre}")
        if hallado:
            dotacion_m3_ha, origen_dot = hallado[0], f"PCR declarado: {hallado[1]}"
        elif estado_res and estado_res.get("volumen_hm3") is not None and local is not None:
            pcr = local[local["variable"] == "water_allocation"]
            if len(pcr):
                tabla_pcr = pd.DataFrame({
                    "fecha_inicio": pcr["date"],
                    "cultivo": pcr["product"].str.extract(r"cultivo=([^/]+)")[0].str.strip().fillna(""),
                    "dotacion_m3_ha": pcr["value"]})
                ana = res.dotacion_analoga(tabla_pcr, res.serie_volumen(reservorios),
                                           estado_res["volumen_hm3"], cultivo_id)
                if ana:
                    dotacion_m3_ha = ana["dotacion_m3_ha"]
                    origen_dot = ("escenario: dotación de la campaña análoga "
                                  f"{ana['campania_analoga']} (EX-15)")
        if dotacion_m3_ha is None:
            origen_dot = "por determinar (EX-04)"
    proc["dotacion"] = origen_dot

    origen_int = "parámetro"
    if intervalo_dias is None:
        hallado = _ultimo(local, "turn_interval")
        if hallado:
            intervalo_dias, origen_int = int(hallado[0]), f"turnos declarados: {hallado[1]}"
        else:
            origen_int = "por determinar (EX-16)"
            adv.append("Intervalo de turnos desconocido: se simula sin turnos (secano) salvo que se declare.")
    proc["turnos"] = origen_int

    origen_ef = "parámetro"
    if eficiencia is None:
        hallado = _ultimo(local, "application_efficiency")
        if hallado:
            eficiencia, origen_ef = hallado[0], f"declarada: {hallado[1]}"
        else:
            eficiencia = EFICIENCIA_REFERENCIA["gravedad"]
            origen_ef = "0,5 por defecto (gravedad, POR VALIDAR)"
    proc["eficiencia"] = origen_ef

    riego = Riego(intervalo_dias=intervalo_dias, dotacion_m3_ha=dotacion_m3_ha, eficiencia=eficiencia,
                  modo="dotacion" if dotacion_m3_ha is not None else "llenado")

    dias = c.ciclo_dias or 365
    anios_filtro, info_enso = None, None
    if ctx.enso is not None and not ctx.enso.empty:
        anios = sorted(set(tabla.dropna().index.year))
        clases, ultimo = _clasificar_enso(ctx.enso, inicio, dias, anios)
        info_enso = {"icen_ultimo": ultimo, "clases": clases,
                     "anios_nino": sorted(a for a, k in clases.items() if k == "nino"),
                     "anios_nina": sorted(a for a, k in clases.items() if k == "nina")}
        proc["enso"] = "ICEN oficial (ENFEN/IGP), umbrales +0,4 / −1,0 POR VERIFICAR"
        if condicion_enso:
            elegidos = [a for a, k in clases.items() if k == condicion_enso]
            if len(elegidos) >= 10:
                anios_filtro = elegidos
            else:
                adv.append(f"Solo {len(elegidos)} años '{condicion_enso}' (< 10, EX-07): se usan todos los "
                           f"años; análogos individuales: {elegidos}")

    out = simular_escenarios(cultivo_id, inicio, suelo, riego, tabla, area_ha, metrica, anios_filtro)
    hc = hindcast(tabla, inicio.strftime("%m-%d"), dias, metrica, anios=anios_filtro)
    for r in out["resultados"].values():
        adv.extend(a for a in r.advertencias if a not in adv)
    return Evaluacion(cultivo_id, inicio, suelo, riego, out["resultados"], out["escenarios"],
                      resumen(out["resultados"]), ctx.validacion, hc, estado_res, info_enso, proc, adv)


def evaluar(
    cultivo_id: str,
    inicio: str | pd.Timestamp,
    clima: pd.DataFrame,
    *,
    estacion: pd.DataFrame | None = None,
    nombre_estacion: str = "estación",
    suelo_datos: pd.DataFrame | None = None,
    local: pd.DataFrame | None = None,
    enso: pd.DataFrame | None = None,
    reservorios: pd.DataFrame | None = None,
    lluvia_datos: pd.DataFrame | None = None,
    relieve: pd.DataFrame | None = None,
    area_ha: float = 1.0,
    intervalo_dias: int | None = None,
    dotacion_m3_ha: float | None = None,
    eficiencia: float | None = None,
    condicion_enso: str | None = None,
    metrica: str = "deficit",
    fecha_consulta: str | pd.Timestamp | None = None,
) -> Evaluacion:
    """Atajo: :func:`preparar` + :func:`evaluar_ctx` para un solo cultivo y fecha."""
    ctx = preparar(clima, estacion=estacion, nombre_estacion=nombre_estacion, suelo_datos=suelo_datos,
                   local=local, enso=enso, reservorios=reservorios, fecha_consulta=fecha_consulta,
                   lluvia_datos=lluvia_datos, relieve=relieve)
    return evaluar_ctx(ctx, cultivo_id, inicio, area_ha=area_ha, intervalo_dias=intervalo_dias,
                       dotacion_m3_ha=dotacion_m3_ha, eficiencia=eficiencia,
                       condicion_enso=condicion_enso, metrica=metrica)
