"""Validación con estación, reservorio, hindcast y pipeline de punta a punta."""

import numpy as np
import pandas as pd
import pytest

from nortex_motor import reservorio as res
from nortex_motor.hindcast import hindcast, wilson
from nortex_motor.pipeline import evaluar
from nortex_motor.validacion import (
    estacion_desde_largo,
    factor_lluvia_mensual,
    metricas,
    sesgo_mensual,
    validar_y_calibrar,
)


def _largo(df, prefijo="", fuente="x", producto="p"):
    filas = []
    for col in df.columns:
        for d, v in df[col].items():
            filas.append({"date": d, "aoi_id": "a", "lat": 0.0, "lon": 0.0, "source": fuente,
                          "product": producto, "variable": f"{prefijo}{col}", "value": v,
                          "unit": "", "qc_flag": "ok"})
    return pd.DataFrame(filas)


# --- Validación ------------------------------------------------------------------------

def test_metricas_y_sesgo_mensual():
    idx = pd.date_range("2020-01-01", "2021-12-31")
    obs = pd.Series(30.0, index=idx)
    mod = obs + 2.5
    m = metricas(mod, obs)
    assert m["n"] == len(idx) and m["sesgo"] == pytest.approx(2.5) and m["rmse"] == pytest.approx(2.5)
    assert sesgo_mensual(mod, obs).round(6).eq(2.5).all()


def test_factor_lluvia():
    idx = pd.date_range("2020-01-01", "2021-12-31")
    mod = pd.Series(1.0, index=idx)
    obs = pd.Series(0.5, index=idx)
    assert factor_lluvia_mensual(mod, obs).eq(0.5).all()


def test_calibracion_corrige_sesgo_con_validacion_cruzada(clima_sintetico):
    estacion = clima_sintetico.loc["2010":"2019", ["tmax", "tmin"]].copy()
    estacion["tmax"] -= 2.6  # el modelo es 2,6 °C más cálido en Tmax
    estacion["tmin"] -= 0.5
    corregido, rep = validar_y_calibrar(clima_sintetico, estacion, "estación prueba")
    assert rep["corregido"] and rep["dias_comunes"] == len(estacion)
    assert rep["variables"]["tmax"]["antes"]["sesgo"] == pytest.approx(2.6)
    assert abs(rep["variables"]["tmax"]["despues_cv"]["sesgo"]) < 1e-6
    assert (corregido["tmax"] - clima_sintetico["tmax"]).mean() == pytest.approx(-2.6)
    assert (corregido["et0"] < clima_sintetico["et0"]).all()  # más frío → menos ET0


def test_sin_traslape_no_corrige(clima_sintetico):
    estacion = clima_sintetico.loc["2010-01-01":"2010-03-31", ["tmax"]] - 1
    corregido, rep = validar_y_calibrar(clima_sintetico, estacion)
    assert not rep["corregido"] and corregido.equals(clima_sintetico)


def test_estacion_desde_largo_acepta_ids():
    idx = pd.date_range("2024-01-01", periods=3)
    df = _largo(pd.DataFrame({"tmax": [30.0] * 3, "precip": [0.0] * 3}, index=idx), prefijo="station_")
    est = estacion_desde_largo(df)
    assert list(est.columns) == ["tmax", "precip"]


# --- Reservorio ------------------------------------------------------------------------

def test_estado_reservorio_percentil():
    fechas = [pd.Timestamp(a, 11, 10) for a in range(2015, 2026)]
    vol = pd.Series([100.0 + 5 * i for i in range(len(fechas))], index=fechas)
    vol[pd.Timestamp("2026-11-10")] = 60.0
    e = res.estado(vol.sort_index(), "2026-11-15")
    assert e["volumen_hm3"] == 60 and e["antiguedad_dias"] == 5
    assert e["porcentaje_capacidad"] == pytest.approx(round(100 * 60 / 195.6, 1))
    assert e["percentil"] == 0 and e["categoria"] == "bajo" and e["anios_comparados"] == 11


def test_estado_reservorio_historia_insuficiente():
    vol = pd.Series([186.04, 195.6], index=pd.to_datetime(["2022-05-17", "2022-05-26"]))
    e = res.estado(vol, "2022-06-01")
    assert e["percentil"] is None and e["categoria"] == "por determinar"


def test_dotacion_analoga():
    vol = pd.Series([50.0, 180.0], index=pd.to_datetime(["2020-11-25", "2022-11-25"]))
    pcr = pd.DataFrame({"fecha_inicio": ["2020-12-01", "2022-12-01"], "cultivo": ["arroz", "arroz"],
                        "dotacion_m3_ha": [6000, 12000]})
    a = res.dotacion_analoga(pcr, vol, 170.0, "arroz")
    assert a["dotacion_m3_ha"] == 12000 and a["categoria"] == "escenario"
    assert res.dotacion_analoga(pcr, vol, 170.0, "maiz") is None


def test_serie_volumen_combina_fuentes():
    df = pd.DataFrame([
        {"date": pd.Timestamp("2022-05-26"), "product": "SENAMHI / san_lorenzo",
         "variable": "reservoir_volume_report", "value": 186.04},
        {"date": pd.Timestamp("2022-05-26"), "product": "SENAMHI / poechos",
         "variable": "reservoir_volume_report", "value": 438.3},
        {"date": pd.Timestamp("2026-11-01"), "product": "ANA", "variable": "reservoir_volume",
         "value": 120.0},
    ])
    s = res.serie_volumen(df)
    assert s.tolist() == [186.04, 120.0]


# --- Hindcast -------------------------------------------------------------------------

def test_wilson():
    lo, hi = wilson(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.97


def test_hindcast_tendencia_sale_del_rango(clima_sintetico):
    # El déficit crece cada año: los años nuevos quedan por encima del P90 de los anteriores.
    hc = hindcast(clima_sintetico, "06-01", 80, min_anios=10)
    assert hc["anios_probados"] == 10
    assert hc["cobertura"] < 0.7 and hc["veredicto"].startswith("mal calibrado")


def test_hindcast_clima_estable_calibrado():
    idx = pd.date_range("1991-01-01", "2020-12-31")
    rng = np.random.default_rng(1)
    clima = pd.DataFrame({"tmax": 32.0, "tmin": 20.0,
                          "et0": 5 + rng.normal(0, 0.8, len(idx)), "precip": 0.0}, index=idx)
    hc = hindcast(clima, "06-01", 80, min_anios=10)
    assert hc["cobertura"] >= 0.7


# --- Pipeline -------------------------------------------------------------------------

def test_pipeline_usa_la_mejor_fuente_disponible(clima_sintetico):
    clima_largo = _largo(clima_sintetico.loc["2001":"2020"], producto="POWER")
    estacion = _largo(clima_sintetico.loc["2012":"2019", ["tmax", "tmin"]] - 1.5, prefijo="station_")
    hoy = pd.Timestamp("2020-11-01")
    local = pd.DataFrame([
        {"date": hoy, "product": "lab / textura=franco", "variable": "lab_theta_fc", "value": 0.26},
        {"date": hoy, "product": "lab / textura=franco", "variable": "lab_theta_wp", "value": 0.12},
        {"date": hoy, "product": "Junta / cultivo=caupi", "variable": "water_allocation", "value": 4500},
        {"date": hoy, "product": "Junta / comision=C1", "variable": "turn_interval", "value": 12},
    ])
    ev = evaluar("caupi", "2021-06-01", clima_largo, estacion=estacion, nombre_estacion="prueba",
                 local=local, fecha_consulta=hoy)
    assert ev.suelo.fuente == "laboratorio" and ev.suelo.theta_fc == 0.26
    assert ev.riego.dotacion_m3_ha == 4500 and ev.riego.intervalo_dias == 12
    assert "PCR declarado" in ev.procedencia["dotacion"]
    assert "corregido con prueba" in ev.procedencia["clima"]
    assert ev.validacion["corregido"]
    assert set(ev.resultados) == {"deficit_bajo", "deficit_medio", "deficit_alto"}
    assert ev.hindcast["anios_probados"] > 0
    assert ev.resumen.loc["cobertura_hidrica"].notna().all()


def test_pipeline_sin_datos_locales_declara_vacios(clima_sintetico):
    ev = evaluar("maiz", "2021-06-01", clima_sintetico.loc["2001":"2020"], fecha_consulta="2020-11-01")
    assert ev.procedencia["dotacion"].startswith("por determinar")
    assert ev.procedencia["turnos"].startswith("por determinar")
    assert any("EX-03" in a for a in ev.advertencias)
    assert any("Sin estación" in a for a in ev.advertencias)


def test_no_corrige_si_no_mejora(clima_sintetico):
    # Estación igual al modelo más ruido: corregir no mejora → no se aplica.
    rng = np.random.default_rng(0)
    est = clima_sintetico.loc["2010":"2019", ["tmax"]].copy()
    est["tmax"] += rng.normal(0, 1.0, len(est))
    corregido, rep = validar_y_calibrar(clima_sintetico, est, "ruidosa")
    info = rep["variables"]["tmax"]
    if not info["corregida"]:
        assert corregido["tmax"].equals(clima_sintetico["tmax"])
        assert any("No se corrige tmax" in a for a in rep["advertencias"])


def test_lluvia_con_pocos_meses_no_se_corrige(clima_sintetico):
    est = clima_sintetico.loc["2010":"2019", ["tmax", "precip"]].copy()
    est["tmax"] -= 2.0
    est["precip"] = est["precip"].where(est.index.day_of_year < 40)  # casi todo faltante
    corregido, rep = validar_y_calibrar(clima_sintetico, est)
    assert rep["variables"]["tmax"]["corregida"]
    assert not rep["variables"]["precip"]["corregida"]
    assert corregido["precip"].equals(clima_sintetico["precip"])
    assert rep["factor_mensual_lluvia"] is None


def test_aviso_reservorio_antiguo(clima_sintetico):
    reservorios = pd.DataFrame([{"date": pd.Timestamp("2018-05-26"), "product": "SENAMHI / san_lorenzo",
                                 "variable": "reservoir_volume_report", "value": 186.0}])
    ev = evaluar("caupi", "2021-06-01", clima_sintetico.loc["2001":"2020"], reservorios=reservorios,
                 intervalo_dias=10, dotacion_m3_ha=4000, fecha_consulta="2020-11-01")
    assert any("no describe la situación actual" in a for a in ev.advertencias)
