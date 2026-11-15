"""Curva de coeficiente de cultivo Kc (FAO-56, enfoque de Kc único, cap. 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nortex_motor.catalogo import Cultivo

ETAPAS = ("inicial", "desarrollo", "media", "final")


def curva_kc(cultivo: Cultivo, inicio: str | pd.Timestamp, dias: int | None = None) -> pd.DataFrame:
    """Kc diario y etapa fenológica desde ``inicio`` (siembra o inicio de campaña).

    Anuales: tramo constante Kc_ini, rampa lineal a Kc_med, tramo constante Kc_med y rampa
    lineal a Kc_fin (FAO-56, fig. 25). Perennes: Kc constante durante ``dias`` (365 por
    defecto) y etapa ``"perenne"``.

    Devuelve un DataFrame indexado por fecha con columnas ``kc`` y ``etapa``.
    """
    inicio = pd.Timestamp(inicio).normalize()
    if cultivo.perenne:
        n = dias or 365
        idx = pd.date_range(inicio, periods=n)
        return pd.DataFrame({"kc": float(cultivo.get("kc_constante")), "etapa": "perenne"},
                            index=idx)

    l_ini, l_des, l_med, l_fin = (int(x) for x in cultivo.get("etapas_dias"))
    k_ini, k_med, k_fin = (float(cultivo.get(k)) for k in ("kc_ini", "kc_med", "kc_fin"))
    kc = np.concatenate([
        np.full(l_ini, k_ini),
        k_ini + (k_med - k_ini) * (np.arange(1, l_des + 1) / l_des),
        np.full(l_med, k_med),
        k_med + (k_fin - k_med) * (np.arange(1, l_fin + 1) / l_fin),
    ])
    etapa = np.repeat(ETAPAS, [l_ini, l_des, l_med, l_fin])
    idx = pd.date_range(inicio, periods=len(kc))
    out = pd.DataFrame({"kc": kc, "etapa": etapa}, index=idx)
    return out.iloc[:dias] if dias else out
