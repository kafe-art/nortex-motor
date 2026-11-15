"""Evaluación de punta a punta con datos reales: POWER + estación + SoilGrids + ICEN + reservorio.

Requiere `nortex_data` (repositorio del equipo) y el histórico de POWER en formato largo:

    python scripts/demo_pipeline.py <power_1991_2026.parquet> <carpeta_repo_nortex_data>
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from nortex_motor.pipeline import evaluar

pd.set_option("display.width", 150)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


def bloque(titulo):
    print(f"\n{'=' * 8} {titulo} {'=' * (90 - len(titulo))}")


def main(ruta_power: str, repo_datos: str) -> None:
    os.chdir(repo_datos)  # config/aois.yaml del módulo de datos
    from nortex_data import get_dataset

    aoi = "parcela_01"
    clima = pd.read_parquet(ruta_power)
    estacion = get_dataset("estacion", aoi, variables=["station_tmax", "station_tmin", "station_precip"])
    suelo = get_dataset("soilgrids", aoi, variables=["soil_theta_fc", "soil_theta_wp"])
    enso = get_dataset("enso", aoi, variables=["icen"])
    reservorios = get_dataset("senamhi_reportes", aoi, variables=["reservoir_volume_report"])

    casos = [
        ("Caupí, sin datos de la Junta", dict()),
        ("Caupí, con PCR y turnos declarados (valores de ejemplo)",
         dict(dotacion_m3_ha=4000, intervalo_dias=14)),
        ("Caupí en años El Niño (ICEN)", dict(dotacion_m3_ha=4000, intervalo_dias=14, condicion_enso="nino")),
    ]
    for titulo, extra in casos:
        ev = evaluar("caupi", "2027-06-01", clima, estacion=estacion,
                     nombre_estacion="NOAA GSOD 84401 (Piura)", suelo_datos=suelo, enso=enso,
                     reservorios=reservorios, **extra)
        bloque(titulo)
        print("Procedencia:")
        for k, v in ev.procedencia.items():
            print(f"  {k:11s} {v}")
        print(f"Suelo: θ_FC {ev.suelo.theta_fc:.3f}, θ_WP {ev.suelo.theta_wp:.3f} ({ev.suelo.fuente})")
        print("Escenarios:", "; ".join(e.etiqueta for e in ev.escenarios))
        print(ev.resumen.to_string())
        hc = ev.hindcast
        print(f"Hindcast §6.5: {hc['aciertos']}/{hc['anios_probados']} años dentro de P10–P90 "
              f"({hc['cobertura']:.0%}, IC95 {hc['ic95'][0]:.0%}–{hc['ic95'][1]:.0%}) → {hc['veredicto']}")
        print("Advertencias:")
        for a in ev.advertencias:
            print("  ⚠", a)

    bloque("Validación del clima con la estación (validación cruzada por año)")
    v = ev.validacion
    print(f"{v['estacion']}: {v['dias_comunes']} días comunes, años {v['anios'][0]}–{v['anios'][-1]}")
    for var, info in v["variables"].items():
        a, d = info["antes"], info["despues_cv"]
        print(f"  {var:6s} sesgo {a['sesgo']:+.2f} → {d['sesgo']:+.2f} | MAE {a['mae']:.2f} → {d['mae']:.2f} "
              f"| r {a['r']:.2f} → {d['r']:.2f}  (n={a['n']})")
        if "mensual_antes" in info:
            ma, md = info["mensual_antes"], info["mensual_despues_cv"]
            print(f"         lluvia mensual: sesgo {ma['sesgo']:+.1f} → {md['sesgo']:+.1f} mm | "
                  f"MAE {ma['mae']:.1f} → {md['mae']:.1f} mm (n={ma['n']} meses)")
    print("  Sesgo mensual Tmax (°C):", v["sesgo_mensual_tmax"])

    bloque("Reservorio San Lorenzo")
    print(ev.reservorio)
    bloque("ENSO")
    print("ICEN último:", ev.enso["icen_ultimo"])
    print("Años de campaña El Niño:", ev.enso["anios_nino"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
