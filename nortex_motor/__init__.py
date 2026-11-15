"""nortex_motor: motor físico de NORTEX (DOC_AVANCE_SIS_v0.7 §6 y §9).

Ejemplo::

    from nortex_motor import Riego, Suelo, simular_escenarios
    from nortex_motor.clima import desde_largo

    clima = desde_largo(datos_power)            # salida larga de nortex_data
    out = simular_escenarios("caupi", "2027-06-01", Suelo(0.12, 0.04),
                             Riego(intervalo_dias=14, dotacion_m3_ha=4000), clima)
"""

from nortex_motor.catalogo import cargar_cultivos, cultivo
from nortex_motor.riego import Riego
from nortex_motor.simulacion import resumen, simular_campania, simular_escenarios
from nortex_motor.suelo import TEXTURAS_FAO56, Suelo
from nortex_motor.valores import Valor

__version__ = "0.1.0"

__all__ = [
    "TEXTURAS_FAO56", "Riego", "Suelo", "Valor", "__version__", "cargar_cultivos", "cultivo",
    "resumen", "simular_campania", "simular_escenarios",
]
