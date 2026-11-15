"""Demostración del motor físico con datos reales de NASA POWER en San Lorenzo.

Uso::

    python scripts/demo_san_lorenzo.py <archivo_power_largo.parquet>

El archivo es la salida de ``nortex_data`` (formato largo) para el punto de referencia.
"""

from __future__ import annotations

import sys

import pandas as pd

from nortex_motor import Riego, resumen, simular_escenarios
from nortex_motor.clima import desde_largo
from nortex_motor.suelo import TEXTURAS_FAO56
from nortex_motor.termico import conteo_degenerado, noches_frescas_por_anio, tmin_media_invierno

pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


def main(ruta: str) -> None:
    clima = desde_largo(pd.read_parquet(ruta))
    print(f"Clima: {clima.index.min().date()} → {clima.index.max().date()} "
          f"({len(clima)} días, ET0 media {clima.et0.mean():.2f} mm/día)\n")

    # Caupí sembrado el 1 de junio tras el arroz; turno cada 14 días; dotación 4 000 m³/ha.
    riego = Riego(intervalo_dias=14, dotacion_m3_ha=4000, eficiencia=0.5)
    for textura in ("arenoso", "franco"):
        out = simular_escenarios("caupi", "2027-06-01", TEXTURAS_FAO56[textura], riego, clima)
        print(f"=== Caupí, siembra 1-jun, suelo {textura}, turno 14 d, 4 000 m³/ha "
              f"(robustez de escenarios: {out['robustez']}, {len(out['ventanas'])} años) ===")
        for e in out["escenarios"]:
            print("  ", e.etiqueta, f"= {e.valor_percentil:.0f} mm")
        print(resumen(out["resultados"]).to_string(), "\n")
        medio = out["resultados"]["medio"]
        for a in medio.advertencias:
            print("  ⚠", a)
        print()

    # Mango: noches frescas (Tmin < 16 °C, jun–ago) y Tmin media del invierno por año.
    nf = noches_frescas_por_anio(clima["tmin"], 16.0)
    tm = tmin_media_invierno(clima["tmin"])
    tabla = pd.DataFrame({"noches_tmin_16": nf, "tmin_media_jun_ago": tm})
    tabla["anomalia_c"] = tabla["tmin_media_jun_ago"] - tabla["tmin_media_jun_ago"].median()
    print("=== Mango: invierno de inducción (jun–ago), POWER; umbral 16 °C POR VALIDAR ===")
    print(tabla.to_string())
    if conteo_degenerado(nf):
        print("  ⚠ El conteo de noches < 16 °C no discrimina entre años (EX-13): se usa la "
              "anomalía de Tmin media como indicador principal.")
    calidos = tabla["anomalia_c"].nlargest(4)
    print("Inviernos más cálidos:", ", ".join(f"{a} (+{v:.2f} °C)" for a, v in calidos.items()))


if __name__ == "__main__":
    main(sys.argv[1])
