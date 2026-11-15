"""Valor etiquetado: toda cifra del motor lleva unidad, categoría y fuente (v0.7 §6.1, §6.7)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: Categorías de dato (enum único de la v0.7 §6.1).
CATEGORIAS = ("observado", "modelado", "pronosticado", "escenario", "simulado", "declarado")


@dataclass(frozen=True)
class Valor:
    """Cifra con su unidad, categoría de dato y fuente."""

    valor: float | int | str | None
    unidad: str
    categoria: str
    fuente: str
    nota: str = ""

    def __post_init__(self) -> None:
        if self.categoria not in CATEGORIAS:
            raise ValueError(f"Categoría inválida {self.categoria!r}; use una de {CATEGORIAS}")

    def a_dict(self) -> dict:
        """Representación serializable (JSON)."""
        return asdict(self)

    def __str__(self) -> str:
        v = "por determinar" if self.valor is None else (
            f"{self.valor:.2f}" if isinstance(self.valor, float) else str(self.valor)
        )
        return f"{v} {self.unidad} ({self.categoria}; {self.fuente})"
