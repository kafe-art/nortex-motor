"""Escenarios, simulación de campaña y adaptador de clima."""

import pandas as pd
import pytest

from nortex_motor import Riego, Suelo, resumen, simular_campania, simular_escenarios
from nortex_motor import escenarios as esc
from nortex_motor.clima import desde_largo
from nortex_motor.suelo import TEXTURAS_FAO56
from nortex_motor.valores import CATEGORIAS, Valor

ARENOSO = TEXTURAS_FAO56["arenoso"]
FRANCO = TEXTURAS_FAO56["franco"]


def test_escenarios_eligen_anios_analogos(clima_sintetico):
    tabla = esc.ventanas_historicas(clima_sintetico, "06-01", 80)
    assert len(tabla) == 20
    e = esc.seleccionar(tabla, "deficit")
    assert [x.nombre for x in e] == ["deficit_bajo", "deficit_medio", "deficit_alto"]
    # el déficit crece con el año: P10 → años tempranos, P90 → años tardíos
    assert e[0].anio_analogo < e[1].anio_analogo < e[2].anio_analogo
    assert esc.robustez(tabla) == "normal"
    assert esc.robustez(tabla.iloc[:10]) == "limitada"


def test_serie_escenario_refechada(clima_sintetico):
    s = esc.serie_escenario(clima_sintetico, 2010, "2027-06-01", 80)
    assert s.index[0] == pd.Timestamp("2027-06-01") and len(s) == 80
    assert s["et0"].iloc[0] == clima_sintetico.loc["2010-06-01", "et0"]
    with pytest.raises(ValueError):
        esc.serie_escenario(clima_sintetico, 2020, "2027-12-01", 80)


def test_campania_valores_etiquetados(clima_sintetico):
    r = simular_campania("caupi", "2010-06-01", FRANCO,
                         Riego(intervalo_dias=10, dotacion_m3_ha=3500), clima_sintetico)
    assert all(isinstance(v, Valor) and v.categoria in CATEGORIAS for v in r.valores.values())
    assert abs(r.valores["cierre_balance"].valor) < 0.1
    assert r.valores["cobertura_hidrica"].nota in {"Cubierto", "No cubierto"}
    assert r.fin - r.inicio == pd.Timedelta(days=79)
    assert any("POR VALIDAR" in a for a in r.advertencias)


def test_suelo_arenoso_sufre_mas_que_franco(clima_sintetico):
    riego = Riego(intervalo_dias=14, modo="llenado")
    a = simular_campania("maiz", "2010-06-01", ARENOSO, riego, clima_sintetico)
    f = simular_campania("maiz", "2010-06-01", FRANCO, riego, clima_sintetico)
    assert a.valores["dias_estres"].valor > f.valores["dias_estres"].valor
    assert a.valores["autonomia_entre_turnos"].valor < f.valores["autonomia_entre_turnos"].valor
    assert a.valores["estres_por_turno"].valor > 0


def test_sin_dotacion_cobertura_por_determinar(clima_sintetico):
    r = simular_campania("caupi", "2010-06-01", FRANCO, Riego(intervalo_dias=7), clima_sintetico)
    assert r.valores["cobertura_hidrica"].valor is None
    assert r.valores["cobertura_hidrica"].nota == "Por determinar"
    assert any("EX-04" in a for a in r.advertencias)


def test_secano_genera_estres_maximo(clima_sintetico):
    r = simular_campania("caupi", "2010-06-01", FRANCO, Riego(intervalo_dias=None),
                         clima_sintetico)
    assert r.valores["riego_neto"].valor == 0
    assert r.valores["ks_minimo"].valor < 0.5
    assert any("secano" in a for a in r.advertencias)


def test_arroz_anegado_usa_demanda_con_perdidas(clima_sintetico):
    r = simular_campania("arroz", "2010-01-01", FRANCO,
                         Riego(intervalo_dias=5, dotacion_m3_ha=12_000), clima_sintetico)
    assert "eta_total" not in r.valores  # el balance de suelo no aplica
    etc = r.valores["etc_total"].valor
    assert r.valores["demanda_neta"].valor > etc  # incluye 4 mm/día de pérdidas
    assert any("anegado" in a for a in r.advertencias)


def test_mango_perenne_un_anio(clima_sintetico):
    r = simular_campania("mango", "2010-01-01", FRANCO, Riego(intervalo_dias=15, modo="llenado"),
                         clima_sintetico)
    assert len(r.diario) == 365
    assert "gdd" not in r.valores


def test_salinidad_maas_hoffman(clima_sintetico):
    salino = Suelo(0.25, 0.12, ece=6.0)
    r = simular_campania("maiz", "2010-06-01", salino, Riego(intervalo_dias=7, modo="llenado"),
                         clima_sintetico)
    assert r.valores["rendimiento_relativo_salinidad"].valor == pytest.approx(100 - 12 * (6 - 1.7))
    r = simular_campania("mango", "2010-01-01", salino, Riego(intervalo_dias=7, modo="llenado"),
                         clima_sintetico)
    assert r.valores["rendimiento_relativo_salinidad"].valor is None
    assert "supera" in r.valores["rendimiento_relativo_salinidad"].nota


def test_simular_escenarios_y_resumen(clima_sintetico):
    out = simular_escenarios("caupi", "2027-06-01", FRANCO,
                             Riego(intervalo_dias=10, dotacion_m3_ha=3500), clima_sintetico)
    assert set(out["resultados"]) == {"deficit_bajo", "deficit_medio", "deficit_alto"}
    tabla = resumen(out["resultados"])
    assert tabla.loc["et0_total", "deficit_bajo"] < tabla.loc["et0_total", "deficit_alto"]
    assert all(r.valores["clima"].categoria == "escenario" for r in out["resultados"].values())


@pytest.mark.parametrize("precip_id", ["precip_power", "rain_power"])
def test_adaptador_acepta_ambos_ids(precip_id):
    fechas = pd.date_range("2024-01-01", periods=2)
    filas = [{"date": d, "variable": v, "value": x}
             for d in fechas for v, x in
             [("tmax", 33), ("tmin", 21), ("et0", 6), (precip_id, 0.5), ("solar", 20)]]
    clima = desde_largo(pd.DataFrame(filas))
    assert list(clima.columns) == ["tmax", "tmin", "et0", "precip"]
    assert clima["precip"].tolist() == [0.5, 0.5]
    with pytest.raises(KeyError):
        desde_largo(pd.DataFrame([{"date": fechas[0], "variable": "tmax", "value": 1}]))
