"""Lectura del catálogo agronómico (`config/cultivos.yaml`)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

ESTADOS = {"POR_VALIDAR", "VALIDADO"}


@dataclass(frozen=True)
class Parametro:
    """Parámetro agronómico con su fuente y estado de validación."""

    valor: Any
    fuente: str
    estado: str


@dataclass(frozen=True)
class Cultivo:
    """Cultivo del catálogo."""

    id: str
    nombre: str
    cientifico: str
    tipo: str  # anual | perenne
    anegado: bool
    parametros: dict[str, Parametro]

    @property
    def perenne(self) -> bool:
        return self.tipo == "perenne"

    def get(self, nombre: str) -> Any:
        """Valor de un parámetro (``None`` si no existe o no tiene valor)."""
        p = self.parametros.get(nombre)
        return None if p is None else p.valor

    def fuente(self, nombre: str) -> str:
        """Fuente y estado de un parámetro, para citarlos en las salidas."""
        p = self.parametros.get(nombre)
        return "sin dato" if p is None else f"{p.fuente}; {p.estado}"

    @property
    def ciclo_dias(self) -> int | None:
        etapas = self.get("etapas_dias")
        return int(sum(etapas)) if etapas else None


def _config_dir() -> Path:
    env = os.getenv("NORTEX_MOTOR_CONFIG")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[1] / "config"
    if (repo / "cultivos.yaml").exists():
        return repo
    return Path(str(resources.files("nortex_motor.config")))


@lru_cache(maxsize=1)
def cargar_cultivos() -> dict[str, Cultivo]:
    """Carga y valida el catálogo; cada parámetro debe tener valor, fuente y estado."""
    data = yaml.safe_load((_config_dir() / "cultivos.yaml").read_text(encoding="utf-8"))
    out: dict[str, Cultivo] = {}
    for cid, c in data["cultivos"].items():
        params = {}
        for nombre, p in c["parametros"].items():
            if not {"valor", "fuente", "estado"} <= set(p):
                raise ValueError(f"{cid}.{nombre}: faltan valor, fuente o estado")
            if p["estado"] not in ESTADOS:
                raise ValueError(f"{cid}.{nombre}: estado inválido {p['estado']!r}")
            params[nombre] = Parametro(p["valor"], str(p["fuente"]), p["estado"])
        tipo = c["tipo"]
        if tipo not in {"anual", "perenne"}:
            raise ValueError(f"{cid}: tipo inválido {tipo!r}")
        if tipo == "anual" and not params.get("etapas_dias"):
            raise ValueError(f"{cid}: un cultivo anual necesita etapas_dias")
        if tipo == "perenne" and "kc_constante" not in params:
            raise ValueError(f"{cid}: un perenne necesita kc_constante")
        out[cid] = Cultivo(cid, c["nombre"], c["cientifico"], tipo, bool(c.get("anegado")), params)
    return out


def cultivo(cid: str) -> Cultivo:
    """Devuelve un cultivo por id (``arroz``, ``caupi``, ``maiz``, ``mango``, ``limon_sutil``)."""
    cat = cargar_cultivos()
    if cid not in cat:
        raise KeyError(f"Cultivo {cid!r} no existe. Opciones: {sorted(cat)}")
    return cat[cid]
