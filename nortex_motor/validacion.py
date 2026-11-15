"""Validación y corrección de la serie regional (POWER/IMERG) con una estación (v0.7 §19).

Flujo:
1. Se comparan los días comunes del modelo y de la estación (SENAMHI o NOAA GSOD).
2. Se estima un **sesgo mensual aditivo** para Tmax y Tmin y un **factor mensual
   multiplicativo** para la lluvia.
3. Se mide la mejora con **validación cruzada dejando un año fuera**: la corrección de cada
   año se ajusta sin ese año, así la mejora reportada no es un sobreajuste. Cada variable se
   corrige **solo si la validación cruzada muestra mejora** (la lluvia, además, con ≥ 12 meses
   completos en la estación).
4. Se corrige toda la serie histórica y se reescala ET0 con la razón de Hargreaves entre las
   temperaturas corregidas y las originales (POWER no publica los demás términos por separado
   en la tabla del motor; la razón conserva la dependencia de ET0 con la temperatura).

La serie corregida sigue siendo **modelado** (no observado): se etiqueta con la estación usada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Ids de estación aceptados (NOAA GSOD en nortex_data y plantillas SENAMHI).
ALIAS_ESTACION = {
    "tmax": ("station_tmax", "senamhi_tmax", "tmax"),
    "tmin": ("station_tmin", "senamhi_tmin", "tmin"),
    "precip": ("station_precip", "senamhi_precip", "precip"),
}
MIN_DIAS_MES = 30        # días pareados mínimos para estimar el sesgo de un mes
MIN_DIAS_TOTAL = 365     # sin un año de traslape no se corrige
FACTOR_LLUVIA = (0.2, 5.0)
MIN_MESES_LLUVIA = 12  # meses completos en la estación para corregir la lluvia


def estacion_desde_largo(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla diaria ``tmax, tmin, precip`` de la estación desde el formato largo."""
    ancho = df.pivot_table(index="date", columns="variable", values="value", aggfunc="first")
    ancho.index = pd.to_datetime(ancho.index)
    out = {}
    for col, alias in ALIAS_ESTACION.items():
        elegido = next((a for a in alias if a in ancho.columns), None)
        if elegido is not None:
            out[col] = ancho[elegido]
    if not out:
        raise KeyError(f"No hay variables de estación; se buscó {ALIAS_ESTACION}")
    return pd.DataFrame(out).sort_index()


def metricas(modelo: pd.Series, obs: pd.Series) -> dict:
    """Sesgo (modelo − observado), MAE, RMSE y correlación en los días comunes."""
    par = pd.concat({"m": modelo, "o": obs}, axis=1).dropna()
    if par.empty:
        return {"n": 0, "sesgo": None, "mae": None, "rmse": None, "r": None}
    d = par["m"] - par["o"]
    r = par["m"].corr(par["o"]) if len(par) > 2 and par["o"].std() > 0 else None
    return {"n": int(len(par)), "sesgo": float(d.mean()), "mae": float(d.abs().mean()),
            "rmse": float(np.sqrt((d ** 2).mean())), "r": None if r is None else float(r)}


def sesgo_mensual(modelo: pd.Series, obs: pd.Series, min_n: int = MIN_DIAS_MES) -> pd.Series:
    """Sesgo medio por mes (modelo − observado); meses con pocos datos usan el sesgo global."""
    par = pd.concat({"m": modelo, "o": obs}, axis=1).dropna()
    d = (par["m"] - par["o"])
    global_ = d.mean() if len(d) else 0.0
    por_mes = d.groupby(d.index.month).agg(["mean", "count"])
    out = pd.Series(global_, index=range(1, 13), dtype=float)
    ok = por_mes["count"] >= min_n
    out.loc[por_mes.index[ok]] = por_mes.loc[ok, "mean"]
    return out


def factor_lluvia_mensual(modelo: pd.Series, obs: pd.Series, min_n: int = MIN_DIAS_MES) -> pd.Series:
    """Factor obs/modelo de la lluvia total por mes (1 si no hay información suficiente)."""
    par = pd.concat({"m": modelo, "o": obs}, axis=1).dropna()
    out = pd.Series(1.0, index=range(1, 13), dtype=float)
    for mes, g in par.groupby(par.index.month):
        if len(g) >= min_n and g["m"].sum() >= 5.0:
            out.loc[mes] = float(np.clip(g["o"].sum() / g["m"].sum(), *FACTOR_LLUVIA))
    return out


def hargreaves_relativo(tmax: pd.Series, tmin: pd.Series) -> pd.Series:
    """Término de temperatura de Hargreaves: (Tmedia + 17,8) · √(Tmax − Tmin)."""
    return ((tmax + tmin) / 2 + 17.8) * np.sqrt((tmax - tmin).clip(lower=0))


def aplicar(clima: pd.DataFrame, s_tmax: pd.Series | None, s_tmin: pd.Series | None,
            f_lluvia: pd.Series | None) -> pd.DataFrame:
    """Aplica sesgos mensuales y factor de lluvia; reescala ET0 con Hargreaves."""
    out = clima.copy()
    mes = out.index.month
    if s_tmax is not None:
        out["tmax"] = out["tmax"] - s_tmax.reindex(mes).to_numpy()
    if s_tmin is not None:
        out["tmin"] = out["tmin"] - s_tmin.reindex(mes).to_numpy()
    if f_lluvia is not None:
        out["precip"] = out["precip"] * f_lluvia.reindex(mes).to_numpy()
    if s_tmax is not None or s_tmin is not None:
        antes = hargreaves_relativo(clima["tmax"], clima["tmin"])
        despues = hargreaves_relativo(out["tmax"], out["tmin"])
        razon = (despues / antes).replace([np.inf, -np.inf], np.nan).clip(0.5, 1.5).fillna(1.0)
        out["et0"] = out["et0"] * razon
    return out


def _ajustar(clima, est, anios=None):
    sel = (lambda s: s[s.index.year.isin(anios)]) if anios is not None else (lambda s: s)
    s_tx = sesgo_mensual(sel(clima["tmax"]), sel(est["tmax"])) if "tmax" in est else None
    s_tn = sesgo_mensual(sel(clima["tmin"]), sel(est["tmin"])) if "tmin" in est else None
    f_p = factor_lluvia_mensual(sel(clima["precip"]), sel(est["precip"])) if "precip" in est else None
    return s_tx, s_tn, f_p


def validar_y_calibrar(clima: pd.DataFrame, estacion: pd.DataFrame,
                       nombre_estacion: str = "estación") -> tuple[pd.DataFrame, dict]:
    """Valida la serie regional contra la estación y la corrige si hay traslape suficiente.

    Devuelve ``(clima_corregido, reporte)``. El reporte tiene, por variable, las métricas
    antes y después de corregir (validación cruzada por año), el traslape y la decisión.
    """
    comunes = clima.index.intersection(estacion.index)
    anios = sorted(set(comunes.year))
    reporte = {"estacion": nombre_estacion, "dias_comunes": int(len(comunes)),
               "anios": anios, "variables": {}, "corregido": False, "advertencias": []}
    if len(comunes) < MIN_DIAS_TOTAL or len(anios) < 2:
        reporte["advertencias"].append(
            f"Traslape insuficiente con {nombre_estacion} ({len(comunes)} días): no se corrige.")
        for v in estacion.columns:
            reporte["variables"][v] = {"antes": metricas(clima[v], estacion[v])}
        return clima, reporte

    # Validación cruzada: cada año se corrige con los sesgos estimados en los demás años.
    cv = []
    for anio in anios:
        otros = [a for a in anios if a != anio]
        s_tx, s_tn, f_p = _ajustar(clima, estacion, otros)
        bloque = clima[clima.index.year == anio]
        cv.append(aplicar(bloque, s_tx, s_tn, f_p))
    corregido_cv = pd.concat(cv).sort_index()

    aplicar_var = {}
    for v in estacion.columns:
        info = {"antes": metricas(clima[v], estacion[v]),
                "despues_cv": metricas(corregido_cv[v], estacion[v])}
        if v == "precip":  # en lluvia importa más el total mensual que el día exacto
            mens = lambda s: s.resample("MS").sum(min_count=20)  # noqa: E731
            info["mensual_antes"] = metricas(mens(clima[v]), mens(estacion[v]))
            info["mensual_despues_cv"] = metricas(mens(corregido_cv[v]), mens(estacion[v]))
            n_meses = info["mensual_antes"]["n"]
            mejora = (n_meses >= MIN_MESES_LLUVIA
                      and info["mensual_despues_cv"]["mae"] < info["mensual_antes"]["mae"]
                      and info["despues_cv"]["mae"] <= info["antes"]["mae"])
            motivo = (f"solo {n_meses} meses completos en la estación (< {MIN_MESES_LLUVIA})"
                      if n_meses < MIN_MESES_LLUVIA else "la validación cruzada no muestra mejora")
        else:
            mejora = info["despues_cv"]["mae"] < info["antes"]["mae"]
            motivo = "la validación cruzada no muestra mejora"
        info["corregida"] = bool(mejora)
        if not mejora:
            info["motivo"] = motivo
            reporte["advertencias"].append(f"No se corrige {v} con {nombre_estacion}: {motivo}.")
        aplicar_var[v] = bool(mejora)
        reporte["variables"][v] = info

    s_tx, s_tn, f_p = _ajustar(clima, estacion)
    s_tx = s_tx if aplicar_var.get("tmax") else None
    s_tn = s_tn if aplicar_var.get("tmin") else None
    f_p = f_p if aplicar_var.get("precip") else None
    reporte["sesgo_mensual_tmax"] = None if s_tx is None else s_tx.round(2).to_dict()
    reporte["sesgo_mensual_tmin"] = None if s_tn is None else s_tn.round(2).to_dict()
    reporte["factor_mensual_lluvia"] = None if f_p is None else f_p.round(2).to_dict()
    reporte["corregido"] = any(aplicar_var.values())
    return aplicar(clima, s_tx, s_tn, f_p), reporte
