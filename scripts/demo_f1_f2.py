"""F1 y F2 con datos reales del valle (POWER + estación + SoilGrids + ICEN + reportes SENAMHI).

    python scripts/demo_f1_f2.py <power_1991_2026.parquet> <carpeta_repo_nortex_data>

La dotación por cultivo y los turnos son VALORES DE EJEMPLO hasta tener el PCR real.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from nortex_motor.decision import f1_ventanas, f2_cultivos
from nortex_motor.pipeline import preparar

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

PCR_EJEMPLO = {"arroz": 12000, "caupi": 4000, "maiz": 7000, "mango": 9000, "limon_sutil": 9000}


def local_ejemplo() -> pd.DataFrame:
    hoy = pd.Timestamp("2026-11-15")
    filas = [{"date": hoy, "product": f"EJEMPLO / cultivo={c}", "variable": "water_allocation", "value": v}
             for c, v in PCR_EJEMPLO.items()]
    filas.append({"date": hoy, "product": "EJEMPLO / comision=San Lorenzo", "variable": "turn_interval",
                  "value": 14})
    return pd.DataFrame(filas)


def _opcional(get_dataset, fuente, aoi, variables):
    """IMERG y SRTM mejoran el riesgo por exceso de lluvia, pero el demo funciona sin ellos."""
    try:
        return get_dataset(fuente, aoi, variables=variables, start="2001-01-01", historico=True)
    except Exception as error:  # sin red o sin credenciales
        print(f"  ({fuente} no disponible: {error})")
        return None


def main(ruta_power: str, repo_datos: str) -> None:
    os.chdir(repo_datos)
    from nortex_data import get_dataset

    aoi = "parcela_01"
    ctx = preparar(
        pd.read_parquet(ruta_power),
        estacion=get_dataset("estacion", aoi, variables=["station_tmax", "station_tmin", "station_precip"]),
        nombre_estacion="NOAA GSOD 84401 (Piura)",
        suelo_datos=get_dataset("soilgrids", aoi, variables=["soil_theta_fc", "soil_theta_wp"]),
        enso=get_dataset("enso", aoi, variables=["icen"]),
        reservorios=get_dataset("senamhi_reportes", aoi, variables=["reservoir_volume_report"]),
        local=local_ejemplo(),
        lluvia_datos=_opcional(get_dataset, "imerg", aoi, ["precipitation"]),
        relieve=_opcional(get_dataset, "srtm", aoi, ["slope"]),
    )
    print("Lluvia para extremos:", ctx.procedencia.get("lluvia_extremos", "POWER (IMERG no disponible)"))

    print("\n======== F1 — ¿Cuándo sembrar caupí? (ventanas de mayo a octubre de 2027) ========")
    f1 = f1_ventanas(ctx, "caupi", "2027-05-01", "2027-10-31", paso_dias=30)
    print(f1["tabla"].to_string(index=False))
    print("Orden:", f1["regla_orden"])
    mejor = f1["decisiones"][[d.inicio.date() for d in f1["decisiones"]].index(f1["tabla"].iloc[0]["inicio"])]
    print(f"\nDetalle de la primera ventana ({mejor.inicio.date()}):")
    print("  Viabilidad:", mejor.viabilidad["estado"], "—", " ".join(mejor.viabilidad["motivos"]))
    for r in mejor.riesgo:
        print(f"  Riesgo [{r['nivel']}] {r['factor']}: {r['evidencia']}")
    print("  Clima de cada escenario (resultados independientes, no se suman):")
    print(pd.DataFrame(mejor.clima_escenarios).to_string(index=False))
    print("  Nota:", mejor.evaluacion.escenarios[0].nota)
    fund = mejor.fundamentacion
    print("  Fundamentación:", fund["nivel"], "; faltan:", "; ".join(fund["faltantes"]))

    print("\n======== F2 — ¿Qué sembrar el 1 de enero de 2027 tras dos campañas de arroz? ========")
    f2 = f2_cultivos(ctx, ["arroz", "caupi", "maiz", "mango", "limon_sutil"], "2027-01-01",
                     historial=["arroz", "arroz"])
    print(f2["tabla"].to_string(index=False))
    for d in f2["decisiones"]:
        print(f"\n  {d.cultivo}: {d.viabilidad['estado']} — {' '.join(d.viabilidad['motivos'])}")
        for regla in d.viabilidad.get("reglas_transicion", []):
            print(f"    · {regla['regla']} ({regla['tipo']}): {regla['texto']}")
        for r in d.riesgo:
            if r["nivel"] in {"alto", "medio"}:
                print(f"    Riesgo [{r['nivel']}] {r['factor']}: {r['evidencia']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
