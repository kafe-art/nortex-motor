# nortex-motor

Motor físico de **NORTEX** (NASA Space Apps 2026, valle de San Lorenzo, Tambogrande, Piura).
Implementa las secciones §6 y §9 de `DOC_AVANCE_SIS_v0.7.md`: a partir de una serie diaria de
clima, el catálogo de cultivos y lo que declara el agricultor (suelo, turnos, dotación),
simula una campaña y devuelve cifras **etiquetadas** con unidad, categoría y fuente.

No depende de ningún repositorio de datos: recibe una tabla diaria `tmax, tmin, et0, precip`.
`nortex_motor.clima.desde_largo()` la construye desde la salida de `nortex_data` y acepta los
dos juegos de ids que existen en el equipo (`precip_power` / `rain_power`).

## Qué calcula

| Módulo | Contenido | v0.7 |
| :-- | :-- | :-- |
| `catalogo` | 5 cultivos (arroz, caupí, maíz, mango, limón sutil); cada parámetro con fuente y estado `POR_VALIDAR` | §5 |
| `kc` | Curva Kc diaria por etapas (FAO-56, Kc único); Kc constante en perennes | §9.1 |
| `balance` | Balance diario FAO-56 en agotamiento con **ETc,adj = Ks·Kc·ET0**, percolación y prueba de cierre de masa | §9.3 |
| `riego` | Turnos, reparto de la dotación, **cobertura hídrica C** y **autonomía entre turnos** | §9.4, §9.5 |
| `termico` | GDD con techo, días de calor en la etapa media, **noches frescas** y **Tmin media del invierno** para el mango | §9.2, §9.6 |
| `escenarios` | Ventana de campaña en cada año histórico, P10/P50/P90 y **año análogo**; estratificable por años (p. ej. ICEN) | §6.2, §6.3 |
| `simulacion` | Campaña completa y comparación de escenarios; arroz anegado como demanda + pérdidas | §14 |

Categorías de dato (enum único §6.1): `observado, modelado, pronosticado, escenario, simulado, declarado`.

## Uso

```python
import pandas as pd
from nortex_motor import Riego, resumen, simular_escenarios
from nortex_motor.clima import desde_largo
from nortex_motor.suelo import TEXTURAS_FAO56

clima = desde_largo(pd.read_parquet("power_san_lorenzo_1991_2026.parquet"))
out = simular_escenarios("caupi", "2027-06-01", TEXTURAS_FAO56["franco"],
                         Riego(intervalo_dias=14, dotacion_m3_ha=4000, eficiencia=0.5), clima)
print(resumen(out["resultados"]))
print(out["resultados"]["medio"].valores["cobertura_hidrica"])
```

Demostración con datos reales:

```bash
python scripts/demo_san_lorenzo.py ../nortex-data/data/processed/power_san_lorenzo_1991_2026.parquet
```

## Resultados de la demostración (POWER 1991–2026, punto de referencia)

**Caupí, siembra 1 de junio, turno cada 14 días, 4 000 m³/ha, riego por gravedad (Ea 0,5):**

| | Favorable (P10, 2000) | Medio (P50, 2007) | Desfavorable (P90, 2015) |
| :-- | --: | --: | --: |
| ET0 de la campaña (mm) | 460 | 498 | 519 |
| Demanda neta (mm) | 344 | 374 | 389 |
| Cobertura hídrica C | 0,58 | 0,53 | 0,51 |
| Autonomía entre turnos, arenoso / franco (días) | 4,3 / 6,9 | 3,9 / 6,3 | 3,6 / 5,9 |
| Días de estrés por turno, arenoso / franco | 9,8 / 7,1 | 10,1 / 7,7 | 10,4 / 8,1 |

Con esa dotación **no se cubre la demanda** en ningún escenario: 4 000 m³/ha brutos son 200 mm
netos frente a ~370 mm de demanda. Con turnos de 14 días, el suelo arenoso aguanta 4 días.

**Mango, invierno de inducción (junio–agosto):** el conteo de noches < 16 °C sale 0 en la
mayoría de años con POWER (la celda de ~50 km suaviza las mínimas), así que el motor lo marca
como **degenerado (EX-13)** y usa la **anomalía de Tmin media**. Los inviernos más cálidos
fueron 2026 (+3,6 °C), 1997 (+3,0 °C), 2023 (+2,8 °C) y 2015 (+1,6 °C), coincidentes con años
El Niño.

## Limitaciones (declaradas)

- Parámetros de cultivo **POR VALIDAR**: rangos de FAO-56, Maas–Hoffman y la v0.7, sin fichas INIA.
- RO = 0 y CR = 0 en el balance (sin escorrentía ni napa somera declarada).
- Zr constante durante el ciclo (no crece desde la siembra).
- Demanda neta = Σ máx(0, ETc − P) diaria: conservadora, la lluvia solo compensa el mismo día.
- La métrica de escenarios por defecto es el **déficit climático Σ(ET0 − P)**, no la
  precipitación: en San Lorenzo la lluvia es casi nula y sus percentiles quedan degenerados.
  Es una propuesta: el equipo debe confirmarla frente a §6.2.
- POWER sobreestima Tmax (~+2,6 °C frente al aeropuerto de Piura), lo que eleva ET0.
- Escenarios: no son pronósticos ni llevan probabilidad.
- **Pendiente:** motor de evaluación F1/F2 (cuatro dimensiones), datos de reservorio/PCR, ICEN
  para estratificar y hindcast (§6.5).

## Pruebas

```bash
pip install -e ".[dev]"
pytest
ruff check .
```
