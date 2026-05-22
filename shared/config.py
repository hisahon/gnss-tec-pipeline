# -*- coding: utf-8 -*-
"""Configuração central do pipeline.

Substitui o antigo ``comv.py``. Lê a variável ``DB_ROOT`` do arquivo ``.env``
e expõe helpers para montar os caminhos das subpastas usadas por todas as
etapas do pipeline.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path


place = 'home'
# place = 'inpe'
# place = 'ssd-inpe'
# place = 'ssd-home'

if place == "home":
    DB_ROOT='D:\database'
    
elif place == 'inpe':
    DB_ROOT='/home/hisashi/database'
    
elif place == 'ssd-home':
    DB_ROOT='nao sei ainda'
    
elif place == 'ssd-inpe':
    DB_ROOT='/media/hisashi/ESSD/database'



# Path(DB_ROOT).expanduser()
DB_ROOT = Path(DB_ROOT).expanduser()
if not DB_ROOT or str(DB_ROOT) == ".":
    raise RuntimeError(
        "DB_ROOT não está definido. Crie um arquivo .env a partir de .env.example."
    )


# ---------- helpers de caminho ----------

def crinex_dir(year: int, doy: int) -> Path:
    folder_out = DB_ROOT / "gnss" / "crinex" / f"{year}" / f"{doy:03d}"
    folder_out.mkdir(parents=True, exist_ok=True)
    return folder_out

def rinex_dir(year: int, doy: int) -> Path:
    return DB_ROOT / "gnss" / "rinex" / f"{year}" / f"{doy:03d}"


def ephemeris_dir(year: int, doy: int) -> Path:
    return DB_ROOT / "gnss" / "ephemeris" / f"{year}" / f"{doy:03d}"

def ephemeris_parquet_dir(year: int, doy: int) -> Path:
    return DB_ROOT / "gnss" / "ephemeris_parquet" / f"{year}" / f"{doy:03d}"

def station_positions_dir() -> Path:
    return DB_ROOT / "gnss" / "station_positions"

def storms_SouthAmerica() -> Path:
    return DB_ROOT / "cptec" / "merged" / "c13_2018-2025_SouthAmerica.parquet"

def tec_dir(year: int, doy: int) -> Path:
    return DB_ROOT / "ionosphere" / "tec" / f"{year}" / f"{doy:03d}"

def tec_pc_dir(year: int, doy: int) -> Path:
    return DB_ROOT / "ionosphere" / "tec_pc" / f"{year}" / f"{doy:03d}"

def dtec_pc_dir(year: int, doy: int) -> Path:
    folder_out = DB_ROOT / "ionosphere" / "dtec_pc" / f"{year}" / f"{doy:03d}"
    folder_out.mkdir(parents=True, exist_ok=True)
    return folder_out

def dtec_ipp_dir(year: int, doy: int) -> Path:
    folder_out = DB_ROOT / "ionosphere" / "dtec_ipp" / f"{year}" / f"{doy:03d}"
    folder_out.mkdir(parents=True, exist_ok=True)
    return folder_out

def dtec_keogram_dir(year: int, doy: int) -> Path:
    folder_out = DB_ROOT / "ionosphere" / "keograms" / f"{year}" / f"{doy:03d}"
    folder_out.mkdir(parents=True, exist_ok=True)
    return folder_out

def dtec_keogram_figs_dir() -> Path:
    folder_out = DB_ROOT / "ionosphere" / "keograms_figs"
    folder_out.mkdir(parents=True, exist_ok=True)
    return folder_out

def tec_error_dir(year: int, doy: int) -> Path:
    return DB_ROOT / "ionosphere" / "tec_error" / f"{year}" / f"{doy:03d}"




# ---------- utilidades ----------

def doy_to_date(year: int, doy: int) -> datetime:
    """Converte (ano, dia-do-ano) em ``datetime``."""
    return datetime.strptime(f"{year}-{doy:03d}", "%Y-%j")


def gnss_week(date: datetime) -> int:
    """Calcula o número da semana GNSS para uma dada data."""
    gnss_epoch = datetime(1980, 1, 6)
    return (date - gnss_epoch).days // 7


def setup_logging(level: int = logging.INFO) -> None:
    """Configura o logging para o pipeline inteiro."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
