"""Cuatro dimensiones, F1 y F2."""

import json

import pandas as pd
import pytest

from nortex_motor.decision import (
    decidir,
    f1_ventanas,
    f2_cultivos,
    reglas_transicion,
    tipo_transicion,
)
from nortex_motor.pipeline import preparar

HOY = pd.Timestamp("2020-11-01")


def _local(**valores):
    filas = []
    for variable, (valor, producto) in valores.items():
        filas.append({"date": HOY, "product": producto, "variable": variable, "value": valor})
    return pd.DataFrame(filas)


@pytest.fixture
def ctx(clima_sintetico):
    local = _local(lab_theta_fc=(0.26, "lab"), lab_theta_wp=(0.12, "lab"),
                   turn_interval=(10, "Junta / comision=C1"))
    return preparar(clima_sintetico.loc["2001":"2020"], local=local, fecha_consulta=HOY)


def test_viabilidad_segun_cobertura(ctx):
    escasa = decidir(ctx, "caupi", "2021-06-01", dotacion_m3_ha=1000)
    assert escasa.viabilidad["estado"] == "no viable"
    assert any("RT-06" in m for m in escasa.viabilidad["motivos"])
    holgada = decidir(ctx, "caupi", "2021-06-01", dotacion_m3_ha=20_000)
    assert holgada.viabilidad["estado"] == "viable"
    sin_dato = decidir(ctx, "caupi", "2021-06-01")
    assert sin_dato.viabilidad["estado"] == "por determinar"


def test_rt08_area_no_autorizada(clima_sintetico):
    local = _local(pcr_area_authorized=(0, "Junta / cultivo=arroz"), turn_interval=(7, "Junta"))
    ctx = preparar(clima_sintetico.loc["2001":"2020"], local=local, fecha_consulta=HOY)
    d = decidir(ctx, "arroz", "2021-01-01", dotacion_m3_ha=20_000)
    assert d.viabilidad["estado"] == "no viable" and any("RT-08" in m for m in d.viabilidad["motivos"])


def test_perenne_siempre_con_restricciones(ctx):
    d = decidir(ctx, "mango", "2021-01-01", dotacion_m3_ha=50_000)
    assert d.viabilidad["estado"] == "viable con restricciones"
    assert any("RT-02" in m for m in d.viabilidad["motivos"])


def test_riesgos_con_evidencia(ctx):
    d = decidir(ctx, "maiz", "2021-06-01", dotacion_m3_ha=3000)
    factores = {r["factor"]: r for r in d.riesgo}
    assert "Estrés hídrico en etapa crítica" in factores
    assert all(r["nivel"] in {"bajo", "medio", "alto", "por determinar"} and r["evidencia"]
               for r in d.riesgo)


def test_conveniencia_con_ky_y_sin_datos(ctx):
    sin = decidir(ctx, "caupi", "2021-06-01", dotacion_m3_ha=4000)
    assert sin.conveniencia["estado"] == "por determinar"
    eco = {"caupi": {"rendimiento_t_ha": 1.5, "precio_soles_kg": 4.0, "costo_soles_ha": 3000}}
    con_poca = decidir(ctx, "caupi", "2021-06-01", eco, dotacion_m3_ha=1500)
    con_mucha = decidir(ctx, "caupi", "2021-06-01", eco, dotacion_m3_ha=20_000)
    m_poca = con_poca.conveniencia["margen_soles_ha"]["medio"]
    m_mucha = con_mucha.conveniencia["margen_soles_ha"]["medio"]
    assert m_mucha == pytest.approx(1.5 * 1000 * 4.0 - 3000, rel=0.05)  # casi sin estrés
    assert m_poca < m_mucha  # el estrés reduce el rendimiento (Ky 1,15)
    assert con_poca.conveniencia["rendimiento_relativo"]["medio"] < 1


def test_fundamentacion(ctx, clima_sintetico):
    d = decidir(ctx, "caupi", "2021-06-01", dotacion_m3_ha=4000)
    assert d.fundamentacion["insumos"]["suelo"] == "buena"
    assert d.fundamentacion["nivel"] == "media"  # parámetros POR VALIDAR, sin estación ni economía
    vacio = preparar(clima_sintetico.loc["2001":"2020"], fecha_consulta=HOY)
    d2 = decidir(vacio, "caupi", "2021-06-01")
    assert d2.fundamentacion["nivel"] == "limitada"
    assert any("PCR" in f for f in d2.fundamentacion["faltantes"])


def test_a_dict_es_json(ctx):
    d = decidir(ctx, "caupi", "2021-06-01", dotacion_m3_ha=4000)
    texto = json.dumps(d.a_dict(), default=str)
    assert '"viabilidad"' in texto and '"fundamentacion"' in texto


def test_f1_ordena_ventanas(ctx):
    out = f1_ventanas(ctx, "caupi", "2021-05-01", "2021-08-01", paso_dias=30, dotacion_m3_ha=4000)
    tabla = out["tabla"]
    assert len(tabla) == 4 and {"viabilidad", "riesgos_altos", "fundamentacion"} <= set(tabla.columns)
    orden = tabla["viabilidad"].map({"viable": 0, "viable con restricciones": 1,
                                     "por determinar": 2, "no viable": 3})
    assert orden.is_monotonic_increasing


def test_transiciones_y_reglas(ctx):
    assert tipo_transicion("arroz", "arroz") == "continuidad"
    assert tipo_transicion("arroz", "caupi") == "cambio"
    assert tipo_transicion("arroz", "mango") == "transición"
    assert tipo_transicion("mango", "caupi") == "renovación"
    reglas = reglas_transicion(ctx, ["arroz", "arroz"], "mango")
    assert any(r.startswith("RT-02") for r in reglas) and any(r.startswith("RT-05") for r in reglas)
    assert any("napa" in r for r in reglas)
    assert any(r.startswith("RT-04") for r in reglas_transicion(ctx, ["arroz", "arroz"], "arroz"))
    assert any(r.startswith("RT-01") for r in reglas_transicion(ctx, ["limon_sutil"], "caupi"))


def test_f2_compara_cultivos(ctx):
    out = f2_cultivos(ctx, ["caupi", "maiz", "arroz"], "2021-06-01", historial=["arroz", "arroz"],
                      dotacion_m3_ha=6000)
    tabla = out["tabla"].set_index("cultivo")
    assert tabla.loc["caupi", "transicion"] == "cambio"
    assert tabla.loc["arroz", "transicion"] == "continuidad"
    assert "RT-04" in tabla.loc["arroz", "reglas"]
    # Con la misma dotación, el caupí (menor demanda) cubre más que el maíz
    assert tabla.loc["caupi", "cobertura_medio"] > tabla.loc["maiz", "cobertura_medio"]
