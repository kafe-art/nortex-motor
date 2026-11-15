"""Catálogo, curva Kc, balance FAO-56, riego y térmico."""

import numpy as np
import pandas as pd
import pytest

from nortex_motor.balance import balance_fao56, cierre
from nortex_motor.catalogo import cargar_cultivos, cultivo
from nortex_motor.kc import curva_kc
from nortex_motor.riego import (
    Riego,
    autonomia_dias,
    cobertura_hidrica,
    dias_de_turno,
    dias_estres_por_turno,
    lamina_por_turno,
)
from nortex_motor.suelo import TEXTURAS_FAO56, Suelo
from nortex_motor.termico import gdd, noches_frescas_a_fecha, noches_frescas_por_anio

# --- Catálogo --------------------------------------------------------------------------


def test_catalogo_cinco_cultivos_con_fuente():
    cat = cargar_cultivos()
    assert set(cat) == {"arroz", "caupi", "maiz", "mango", "limon_sutil"}
    for c in cat.values():
        for nombre, p in c.parametros.items():
            assert p.fuente and p.estado in {"POR_VALIDAR", "VALIDADO"}, (c.id, nombre)


def test_ciclos_dentro_de_rangos_v07():
    assert 120 <= cultivo("arroz").ciclo_dias <= 150
    assert 70 <= cultivo("caupi").ciclo_dias <= 90
    assert 110 <= cultivo("maiz").ciclo_dias <= 140
    assert cultivo("mango").perenne and cultivo("limon_sutil").perenne


def test_limon_sutil_especie_correcta():
    assert cultivo("limon_sutil").cientifico == "Citrus × aurantiifolia"


# --- Kc --------------------------------------------------------------------------------


def test_curva_kc_por_etapas():
    c = cultivo("maiz")  # 20/35/40/30 ; 0,30 → 1,20 → 0,35
    kc = curva_kc(c, "2025-01-01")
    assert len(kc) == 125
    assert (kc["kc"].iloc[:20] == 0.30).all()
    assert kc["kc"].iloc[54] == pytest.approx(1.20)  # fin de desarrollo
    assert (kc.loc[kc.etapa == "media", "kc"] == 1.20).all()
    assert kc["kc"].iloc[-1] == pytest.approx(0.35)
    assert kc.loc[kc.etapa == "desarrollo", "kc"].is_monotonic_increasing
    assert list(dict.fromkeys(kc.etapa)) == ["inicial", "desarrollo", "media", "final"]


def test_curva_kc_perenne_constante():
    kc = curva_kc(cultivo("mango"), "2025-01-01")
    assert len(kc) == 365 and kc["kc"].nunique() == 1 and set(kc.etapa) == {"perenne"}


# --- Suelo y balance ------------------------------------------------------------------


def test_taw_raw_mm():
    s = Suelo(0.12, 0.04)
    assert s.taw_mm(1.2) == pytest.approx(96.0)
    assert s.raw_mm(1.2, 0.5) == pytest.approx(48.0)
    with pytest.raises(ValueError):
        Suelo(0.04, 0.12)


def _serie(valor, n=30, inicio="2025-01-01"):
    return pd.Series(float(valor), index=pd.date_range(inicio, periods=n))


def test_balance_ks_segun_fao56():
    # ET0 5, Kc 1, TAW 100, p 0,5 → RAW 50: sin estrés 10 días, luego Ks = (TAW−Dr)/(0,5·TAW)
    df = balance_fao56(_serie(5), _serie(0), _serie(1), taw=100, p=0.5)
    assert (df["ks"].iloc[:11] == 1).all()  # Dr inicial del día 11 = 50 = RAW
    assert df["dr_inicio"].iloc[11] == pytest.approx(55)
    assert df["ks"].iloc[11] == pytest.approx(0.9)
    assert df["dr_fin"].max() <= 100
    assert abs(cierre(df)) < 1e-9


def test_balance_lluvia_genera_percolacion():
    lluvia = _serie(0, 5)
    lluvia.iloc[2] = 40
    df = balance_fao56(_serie(5, 5), lluvia, _serie(1, 5), taw=100, p=0.5)
    assert df["dp"].iloc[2] == pytest.approx(40 - 15)  # Dr previo 10 + ETc 5
    assert df["dr_fin"].iloc[2] == 0
    assert abs(cierre(df)) < 1e-9


def test_balance_llenado_por_turno_evita_estres():
    turnos = dias_de_turno(pd.date_range("2025-01-01", periods=60), Riego(intervalo_dias=7))
    df = balance_fao56(_serie(5, 60), _serie(0, 60), _serie(1, 60), taw=100, p=0.5,
                       turnos_llenado=turnos)
    assert (df["ks"] == 1).all()  # 7 días × 5 mm = 35 mm < RAW 50
    assert df["riego"].sum() == pytest.approx(df["etc_adj"].sum() - df["dr_fin"].iloc[-1])
    assert abs(cierre(df)) < 1e-9


def test_balance_sin_et0_falla():
    et0 = _serie(5, 5)
    et0.iloc[2] = np.nan
    with pytest.raises(ValueError):
        balance_fao56(et0, _serie(0, 5), _serie(1, 5), taw=100, p=0.5)


# --- Riego (ejemplos de la v0.7 §9.4–9.5 y §17.3) --------------------------------------


def test_autonomia_ejemplo_arenoso_v07():
    raw = TEXTURAS_FAO56["arenoso"].raw_mm(1.2, 0.5)  # 0,08 × 1200 × 0,5 = 48 mm
    assert raw == pytest.approx(48)
    aut = autonomia_dias(raw, 0.8 * 5.0)
    assert aut == pytest.approx(12)
    assert dias_estres_por_turno(16, aut) == pytest.approx(4)
    assert dias_estres_por_turno(None, aut) is None


def test_cobertura_ejemplo_v07():
    r = cobertura_hidrica(600, 0.5, 1.0, 10_000)
    assert r["demanda_bruta_m3"] == pytest.approx(12_000)
    assert r["c"] == pytest.approx(10_000 / 12_000)
    assert r["etiqueta"] == "No cubierto"
    assert cobertura_hidrica(600, 0.5, 1.0, None)["etiqueta"] == "Por determinar"


def test_lamina_por_turno_reparte_dotacion():
    fechas = pd.date_range("2025-01-01", periods=30)
    lam = lamina_por_turno(fechas, Riego(intervalo_dias=10, dotacion_m3_ha=4000, eficiencia=0.5))
    assert (lam > 0).sum() == 3
    assert lam.sum() == pytest.approx(4000 / 10 * 0.5)  # 400 mm brutos → 200 mm netos


# --- Térmico ---------------------------------------------------------------------------


def test_gdd_con_techo():
    g = gdd(pd.Series([40.0, 20.0]), pd.Series([20.0, 5.0]), tbase=10, tupper=30)
    assert g.tolist() == [15.0, 2.5]


def test_noches_frescas():
    idx = pd.date_range("2020-01-01", "2023-12-31")
    tmin = pd.Series(20.0, index=idx)
    tmin[(idx.month == 7) & (idx.day <= 10)] = 15.0  # 10 noches frescas por año
    tmin[(idx.year == 2023) & (idx.month == 7)] = 15.0  # 2023: julio completo
    por_anio = noches_frescas_por_anio(tmin, 16.0)
    assert por_anio.loc[2020] == 10 and por_anio.loc[2023] == 31
    estado = noches_frescas_a_fecha(tmin, 16.0, "2023-07-20")
    assert estado["actual"] == 20 and estado["mediana_historica"] == 10
    assert estado["percentil"] == 100.0


def test_tmin_invierno_y_conteo_degenerado():
    from nortex_motor.termico import conteo_degenerado, tmin_media_invierno

    idx = pd.date_range("2020-01-01", "2022-12-31")
    tmin = pd.Series(18.0, index=idx) + (idx.year - 2020) * 0.5
    medias = tmin_media_invierno(tmin)
    assert medias.round(2).tolist() == [18.0, 18.5, 19.0]
    assert conteo_degenerado(pd.Series([0, 0, 0, 0, 1]))
    assert not conteo_degenerado(pd.Series([3, 10, 0, 7, 12]))
