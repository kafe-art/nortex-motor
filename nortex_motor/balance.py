"""Balance hídrico diario de la zona radicular (FAO-56, ec. 85–88; v0.7 §9.3).

Forma en agotamiento:

    Dr,i = Dr,i−1 − (P − RO)i − Ii − CRi + ETc,adj,i + DPi        0 ≤ Dr,i ≤ TAW
    Ks   = 1                                si Dr,i−1 ≤ RAW
    Ks   = (TAW − Dr,i−1) / ((1 − p)·TAW)   si Dr,i−1 > RAW
    ETc,adj = Ks · Kc · ET0

Supuestos del MVP (declarados): RO = 0 (escorrentía despreciable en riego por gravedad en
terreno nivelado), CR = 0 (sin napa somera declarada). Se resta ETc,adj, nunca ETc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOLERANCIA_CIERRE_MM = 0.1


def balance_fao56(
    et0: pd.Series,
    precip: pd.Series,
    kc: pd.Series,
    taw: float,
    p: float,
    riego_neto: pd.Series | None = None,
    turnos_llenado: pd.Series | None = None,
    dr0: float = 0.0,
) -> pd.DataFrame:
    """Simula el agotamiento diario de la zona radicular.

    Parámetros
    ----------
    et0, precip, kc : series diarias alineadas (mm/día, mm/día, adimensional).
    taw : agua total disponible (mm). ``p`` : fracción de agotamiento sin estrés.
    riego_neto : lámina neta aplicada por día (mm), p. ej. turnos con dotación fija.
    turnos_llenado : serie booleana; en los días ``True`` se riega lo necesario para
        volver a capacidad de campo (riego «a demanda» dentro de cada turno).
    dr0 : agotamiento inicial (mm). 0 = suelo a capacidad de campo.

    Devuelve un DataFrame diario con ``et0, kc, etc, ks, etc_adj, precip, riego, dp,
    dr_inicio, dr_fin``.
    """
    if taw <= 0 or not 0 < p < 1:
        raise ValueError("Se requiere TAW > 0 y 0 < p < 1")
    idx = kc.index
    et0 = et0.reindex(idx).to_numpy(float)
    pr = precip.reindex(idx).fillna(0.0).to_numpy(float)
    kcv = kc.to_numpy(float)
    fijo = (riego_neto.reindex(idx).fillna(0.0).to_numpy(float)
            if riego_neto is not None else np.zeros(len(idx)))
    llenar = (turnos_llenado.reindex(idx).fillna(False).to_numpy(bool)
              if turnos_llenado is not None else np.zeros(len(idx), bool))
    if np.isnan(et0).any():
        raise ValueError("ET0 tiene días sin dato en la campaña; complete o recorte la serie")

    raw = p * taw
    n = len(idx)
    out = {k: np.zeros(n) for k in ("etc", "ks", "etc_adj", "riego", "dp", "dr_inicio", "dr_fin")}
    dr = float(np.clip(dr0, 0.0, taw))
    for i in range(n):
        out["dr_inicio"][i] = dr
        riego = fijo[i] + (dr if llenar[i] else 0.0)
        ks = 1.0 if dr <= raw else max(0.0, (taw - dr) / ((1.0 - p) * taw))
        etc = kcv[i] * et0[i]
        etc_adj = ks * etc
        nuevo = dr - pr[i] - riego + etc_adj
        dp = 0.0
        if nuevo < 0.0:  # exceso sobre capacidad de campo → percolación profunda
            dp, nuevo = -nuevo, 0.0
        if nuevo > taw:  # no se puede extraer más allá del punto de marchitez
            etc_adj -= nuevo - taw
            nuevo = taw
        dr = nuevo
        out["etc"][i], out["ks"][i], out["etc_adj"][i] = etc, ks, etc_adj
        out["riego"][i], out["dp"][i], out["dr_fin"][i] = riego, dp, dr
    df = pd.DataFrame(out, index=idx)
    df.insert(0, "et0", et0)
    df.insert(1, "kc", kcv)
    df.insert(5, "precip", pr)
    return df


def cierre(df: pd.DataFrame) -> float:
    """Error de cierre de masa (mm): Σ(P + I − ETc,adj − DP) − (Dr_inicio − Dr_fin).

    Debe ser ≈ 0 (tolerancia ``TOLERANCIA_CIERRE_MM``); se reporta en cada simulación.
    """
    entradas = df["precip"].sum() + df["riego"].sum()
    salidas = df["etc_adj"].sum() + df["dp"].sum()
    cambio = df["dr_inicio"].iloc[0] - df["dr_fin"].iloc[-1]
    return float(entradas - salidas - cambio)
