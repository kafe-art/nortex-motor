"""Riesgo climático por frecuencia histórica."""

import numpy as np
import pandas as pd

from nortex_motor import riesgo_climatico as rc


def _clima(anios=range(2001, 2021), aguacero=None, calor=None):
    idx = pd.date_range(f"{min(anios)}-01-01", f"{max(anios)}-12-31", freq="D")
    rng = np.random.default_rng(0)
    precip = pd.Series(rng.gamma(0.3, 2.0, len(idx)), index=idx)
    tmax = pd.Series(30.0, index=idx)
    for a in aguacero or []:
        precip.loc[f"{a}-02-10":f"{a}-02-14"] = 60.0
    for a in calor or []:
        tmax.loc[f"{a}-03-01":f"{a}-03-05"] = 38.0
    precip.loc[precip.index.month.isin([6, 7, 8, 9, 10, 11])] = 0.0  # estación seca
    return pd.DataFrame({"precip": precip, "tmax": tmax})


def test_frecuencia_y_peor_anio():
    clima = _clima(aguacero=[2005, 2010, 2017, 2019], calor=[2010])
    out = rc.evaluar("caupi", "2021-01-20", clima)
    # Umbral = P80 de los máximos anuales: solo los aguaceros de 300 mm lo superan
    assert out["anios"] == 20 and out["anios_exceso"] == 4
    assert out["frecuencia_exceso"] == 0.20
    assert out["peor_anio_lluvia"]["anio"] in {2005, 2010, 2017, 2019}
    assert out["nivel_exceso"] == "alto"  # sensibilidad alta: 20 % ≥ 15 %
    # Floración del caupí: días 35–55 (24-feb a 16-mar) → el calor de marzo de 2010 cae dentro
    assert out["anios_calor"] == 1 and out["peor_anio_calor"]["anio"] == 2010
    assert "POWER" in out["fuente_lluvia"]


def test_la_frecuencia_depende_de_la_fecha_de_siembra():
    clima = _clima(aguacero=[2005, 2010, 2017, 2019])
    assert rc.evaluar("caupi", "2021-07-01", clima)["anios_exceso"] == 0


def test_sensibilidad_cambia_el_nivel():
    clima = _clima(aguacero=[2005, 2010, 2017, 2019])  # 20 % de los años
    assert rc.evaluar("caupi", "2021-01-20", clima)["nivel_exceso"] == "alto"   # sensibilidad alta
    assert rc.evaluar("arroz", "2021-01-20", clima)["nivel_exceso"] == "medio"  # sensibilidad baja
    niveles = {k: rc._nivel(0.12, c) for k, c in rc.CORTES_EXCESO.items()}
    assert niveles == {"alta": "medio", "media": "medio", "baja": "bajo"}


def test_calor_exige_dias_minimos():
    clima = _clima(calor=[2010])
    clima.loc["2012-03-02", "tmax"] = 38.0  # un solo día caluroso no cuenta
    assert rc.evaluar("caupi", "2021-01-20", clima)["anios_calor"] == 1


def test_usa_imerg_solo_con_registro_largo():
    clima = _clima()
    corta = clima["precip"].loc["2015":]
    larga = clima["precip"] * 1.0
    assert "POWER" in rc.evaluar("maiz", "2021-01-20", clima, corta)["fuente_lluvia"]
    assert "IMERG" in rc.evaluar("maiz", "2021-01-20", clima, larga)["fuente_lluvia"]


def test_sin_ventana_de_floracion():
    out = rc.evaluar("limon_sutil", "2021-01-01", _clima())
    assert out["nivel_calor"] == "por determinar"
    assert out["ventana_floracion"] == "sin dato de floración"


def test_historial_corto_queda_por_determinar():
    # Con 3 años no hay umbral: no puede salir «bajo» con 0 % (hallazgo de la revisión del equipo)
    clima = _clima(anios=range(2018, 2021), aguacero=[2019])
    out = rc.evaluar("caupi", "2021-01-20", clima)
    assert out["umbral_rx5day_mm"] is None
    assert out["nivel_exceso"] == "por determinar" and out["frecuencia_exceso"] is None
    assert "historial insuficiente" in out["motivo_exceso"]
