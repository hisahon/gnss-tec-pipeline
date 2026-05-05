# -*- coding: utf-8 -*-
"""Utilitários compartilhados entre os scripts ``03*.py``.

Cada script da etapa 03 é independente e roda do terminal sozinho, mas
todos compartilham o mesmo padrão de CLI:

    python 03X_*.py --year 2025 --doy 298   # um dia
    python 03X_*.py --year 2025             # ano inteiro
    python 03X_*.py                         # toda a árvore rinex/

Este módulo concentra a lógica de descoberta dos pares ``(year, doy)``
disponíveis em disco e o parser de argumentos comum, evitando
duplicação. Não é executável.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from shared.config import crinex_dir, rinex_dir, tec_dir, tec_pc_dir

log = logging.getLogger("process.utils")


# --------------------------------------------------------------------------
# Descoberta de pares (year, doy)
# --------------------------------------------------------------------------

def _folder_root(referencefolder: str) -> Path:
    """Deduz o diretório raiz de ``rinex/`` a partir de ``config.rinex_dir``.

    A estrutura esperada é ``.../rinex/{year}/{doy}``; portanto subir dois
    níveis a partir de qualquer par (year, doy) retorna a raiz.
    """
    
    referencefolder = referencefolder.lower()
    
    if referencefolder == "crinexreferencefolder":
        return crinex_dir(2000, 1).parent.parent
    
    if referencefolder == "rinexrawreferencefolder":
        return rinex_dir(2000, 1).parent.parent
    
    if referencefolder == "rinexpklreferencefolder":
        return tec_dir(2000, 1).parent.parent
    
    if referencefolder == "tecrawreferencefolder":
        return tec_dir(2000, 1).parent.parent
    
    if referencefolder == "tecpcreferencefolder":
        return tec_pc_dir(2000, 1).parent.parent
    

def discover_year_doys(
    year_filter: int | None = None,
    doy_filter: int | None = None,
    referencefolder: str = "Crinex",
) -> list[tuple[int, int]]:
    """Descobre pares ``(year, doy)`` existentes sob ``rinex/``.

    Filtros (opcionais) restringem o resultado:
      - ``year_filter``: apenas o ano dado.
      - ``doy_filter``: apenas o doy dado (requer ``year_filter``).
    """
    root = _folder_root(referencefolder)
    if not root.exists():
        log.error("Pasta rinex não encontrada: %s", root)
        return []

    pairs: list[tuple[int, int]] = []
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        if year_filter is not None and year != year_filter:
            continue
        for doy_dir in sorted(year_dir.iterdir()):
            if not doy_dir.is_dir() or not doy_dir.name.isdigit():
                continue
            doy = int(doy_dir.name)
            if doy_filter is not None and doy != doy_filter:
                continue
            pairs.append((year, doy))
    return pairs


# --------------------------------------------------------------------------
# CLI comum
# --------------------------------------------------------------------------

def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Cria o parser padrão usado por todos os scripts ``03*.py``."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--year", type=int, help="Filtrar por um ano específico")
    p.add_argument("--doy", type=int, help="Filtrar por um doy (requer --year)")
    return p


def resolve_pairs(args: argparse.Namespace,
                  referencefolder: str) -> list[tuple[int, int]]:
    """Aplica os filtros do CLI e devolve os pares ``(year, doy)`` a processar.

    Valida a combinação ``--doy`` sem ``--year`` (proibida).
    """
    if args.doy is not None and args.year is None:
        raise SystemExit("--doy requer --year.")

    pairs = discover_year_doys(args.year, args.doy, referencefolder)

    if not pairs:
        log.warning(
            "Nenhum par (year, doy) encontrado para year=%s doy=%s",
            args.year, args.doy,
        )
    return pairs
