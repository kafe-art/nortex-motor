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
| `validacion` | Compara POWER con una estación (SENAMHI o NOAA), corrige Tmax/Tmin/lluvia **solo si la validación cruzada por año muestra mejora** y reescala ET0 | §19 |
| `reservorio` | Volumen, % de capacidad útil, percentil frente a otros años y dotación análoga | D-17, EX-15 |
| `hindcast` | ¿El rango P10–P90 habría contenido lo observado? Cobertura con IC de Wilson | §6.5 |
| `decision` | **F1** (`f1_ventanas`) y **F2** (`f2_cultivos`) con **cuatro dimensiones separadas**: viabilidad (RT-06, RT-08, salinidad, compromiso perenne), riesgo (factores con nivel y evidencia), conveniencia (margen ajustado por Ky de FAO-33 y salinidad, margen por m³) y fundamentación (procedencia de cada insumo y datos que faltan, RT-07). Sin índice compuesto: orden lexicográfico declarado | §3, §4, §14 |
| `pipeline` | `preparar()` + `evaluar_ctx()` (o el atajo `evaluar()`): elige la mejor fuente para cada insumo (laboratorio > SoilGrids; PCR > análoga; clima corregido > sin corregir), filtra escenarios por ICEN y registra la procedencia | §14 |

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

## Evaluación de punta a punta con datos reales

```bash
python scripts/demo_pipeline.py power_san_lorenzo_1991_2026.parquet <repo nortex-data>
```

Resultado (punto piloto, caupí 1-jun, 4 000 m³/ha, turnos de 14 días de ejemplo):

| Validación con la estación NOAA de Piura (2016–2025, 2 891 días) | Antes | Después (validación cruzada) |
| :-- | --: | --: |
| Tmax: sesgo / MAE / r | +2,70 °C / 3,02 / 0,57 | 0,00 °C / 1,48 / 0,75 |
| Tmin: sesgo / MAE / r | +0,52 °C / 1,17 / 0,87 | 0,01 °C / 1,01 / 0,87 |
| Lluvia | no se corrige: solo 3 meses completos en la estación | |

- POWER sobreestima Tmax hasta +4 °C entre mayo y noviembre; al corregir, ET0 baja ~15 % y la
  cobertura hídrica pasa de 0,51–0,58 a **0,62–0,74**.
- **Hindcast:** 21 de 26 años dentro de P10–P90 (81 %, IC95 62–91 %) → calibrado.
- Advertencias automáticas: suelo de SoilGrids, volumen del reservorio de 2022 (dato antiguo),
  parámetros por validar.
- La estación está a 33 km, en el aeropuerto: parte de la diferencia puede ser real y no
  error de POWER. Con estaciones SENAMHI del valle (Chulucanas, Partidor) la corrección sería
  más representativa: basta pasar su serie en `estacion=`.

## F1 y F2 con datos reales

```bash
python scripts/demo_f1_f2.py power_san_lorenzo_1991_2026.parquet <repo nortex-data>
```

Con dotaciones y turnos **de ejemplo** (PCR real pendiente), riego por gravedad (Ea 0,5):

- **F1, caupí:** todas las ventanas de mayo a octubre de 2027 salen «no viable» (cobertura 53–65 %).
  Las de fin de mayo y junio tienen menos demanda y menos estrés.
- **F2, 1 de enero de 2027 tras dos campañas de arroz:** ningún cultivo alcanza con esas
  dotaciones. Riesgos altos por El Niño Costero (ICEN +4,50; en años análogos la lluvia de la
  campaña llegó a 1 077–1 350 mm) y por invierno cálido para el mango (+1,4 °C). Reglas RT-02,
  RT-04 y RT-05 activadas según el historial.
- **Sensibilidad:** con riego por goteo (Ea 0,9) y la misma agua, caupí y maíz pasan a «viable»
  (cobertura 112–118 %); el mango sigue en 54–56 %.
- **Calibración pendiente:** el motor estima 28 700 m³/ha brutos para mango (ETc 1 529 mm, ET0 anual
  2 184 mm), muy por encima de los 6 000–9 000 m³/ha de la v0.7. Hay que validar el Kc del mango
  (riego deficitario en invierno, dosel incompleto) y la ET0 anual con INIA o la Junta.

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
- **Pendiente:** datos reales de PCR, turnos, economía de Piura y serie histórica del reservorio;
  calibrar Kc de perennes; F3 (seguimiento con Sentinel-2/ECOSTRESS).

## Pruebas

```bash
pip install -e ".[dev]"
pytest
ruff check .
```
