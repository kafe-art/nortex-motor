"""Suelo declarado y agua disponible (v0.7 §9.3). Todo el balance se expresa en mm."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Suelo:
    """Propiedades hídricas del suelo de la parcela.

    ``theta_fc`` y ``theta_wp`` son humedades volumétricas (m³/m³) a capacidad de campo
    y punto de marchitez. ``ece`` es la conductividad eléctrica del extracto (dS/m).
    ``fuente`` indica el origen: laboratorio, declarado o SoilGrids (dato global, EX-03).
    """

    theta_fc: float
    theta_wp: float
    ece: float | None = None
    textura: str = ""
    fuente: str = "declarado"

    def __post_init__(self) -> None:
        if not 0 < self.theta_wp < self.theta_fc < 1:
            raise ValueError("Se requiere 0 < θ_WP < θ_FC < 1 (m³/m³)")
        if self.ece is not None and self.ece < 0:
            raise ValueError("ECe no puede ser negativa")

    def taw_mm(self, zr_m: float) -> float:
        """Agua total disponible TAW = 1000 · (θ_FC − θ_WP) · Zr  [mm]."""
        return 1000.0 * (self.theta_fc - self.theta_wp) * zr_m

    def raw_mm(self, zr_m: float, p: float) -> float:
        """Agua fácilmente disponible RAW = p · TAW  [mm]."""
        return p * self.taw_mm(zr_m)


#: Valores típicos por textura (FAO-56, tabla 19, punto medio). Solo para demostraciones
#: o cuando no hay análisis de laboratorio; se marcan como dato de literatura.
TEXTURAS_FAO56 = {
    "arenoso": Suelo(0.12, 0.04, textura="arenoso", fuente="FAO-56 T19 (punto medio)"),
    "franco_arenoso": Suelo(0.23, 0.11, textura="franco arenoso", fuente="FAO-56 T19 (punto medio)"),
    "franco": Suelo(0.25, 0.12, textura="franco", fuente="FAO-56 T19 (punto medio)"),
    "franco_limoso": Suelo(0.29, 0.15, textura="franco limoso", fuente="FAO-56 T19 (punto medio)"),
    "arcilloso": Suelo(0.36, 0.22, textura="arcilloso", fuente="FAO-56 T19 (punto medio)"),
}
