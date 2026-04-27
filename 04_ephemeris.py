# -*- coding: utf-8 -*-
"""
Etapa 04 — Processamento de Efemérides e Posições de Satélites.

Este script é responsável por:
1. Consolidar arquivos SP3 em um pickle de órbita única.
2. Interpolar as coordenadas (X, Y, Z) dos satélites para os horários das observações.
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import timedelta
from pathlib import Path

import pandas as pd
import config
import sp3

# Configuração do Log
log = logging.getLogger("ephemeris")

# Regex para identificar o PRN no nome do arquivo (ex: G12)
PRN_RE = re.compile(r"([GRECIJS]\d{2})$")

def list_files(folder: Path, extension: str) -> list[Path]:
    """Retorna recursivamente todos os arquivos com a extensão dada."""
    return sorted(folder.rglob(f"*{extension}"))

def prn_from_path(path: Path) -> str:
    """Extrai o identificador de PRN do nome do arquivo."""
    stem = path.stem
    tok = stem.split("_")[-1]
    m = PRN_RE.search(tok) or PRN_RE.search(stem)
    return m.group(1) if m else tok

def load_or_build_orbit(year: int, doy: int) -> pd.DataFrame:
    """Carrega o SP3 consolidado ou o constrói a partir da base de dados."""
    eph_dir = config.ephemeris_dir(year, doy)
    eph_dir.mkdir(parents=True, exist_ok=True)

    sp3_pickle = eph_dir / f"{year}{doy:03d}_complete.pkl"
    if sp3_pickle.exists():
        log.info("Lendo SP3 consolidado de %s", sp3_pickle)
        return pd.read_pickle(sp3_pickle)

    dt_in = config.doy_to_date(year, doy)
    week = config.gnss_week(dt_in)
    log.info("SP3 não encontrado. Construindo (GNSS week %d)...", week)

    # Concatena efemérides da pasta raiz configurada
    df_orbit = sp3.concat_all(str(config.DB_ROOT / "gnss" / "ephemeris"), dt_in)
    df_orbit.to_pickle(sp3_pickle)
    return df_orbit

def compute_satellite_positions(
    tec_pc_folder: Path,
    df_orbit: pd.DataFrame,
    ref_date: pd.Timestamp,
) -> None:
    """Interpola as coordenadas do satélite e salva nos parquets."""
    df_orbit = df_orbit.copy()
    df_orbit["sec"] = (
        (df_orbit["horario"] - ref_date).dt.total_seconds().astype(int).to_numpy()
    )

    files = list_files(tec_pc_folder, ".parquet")
    total = len(files)

    for i, pq_path in enumerate(files, start=1):
        prn = prn_from_path(pq_path)
        df_orb_prn = (
            df_orbit[df_orbit["prn"] == prn]
            .sort_values("sec")
            .drop_duplicates(subset=["sec"], keep="first")
        )
        
        if df_orb_prn.empty:
            log.warning("[%d/%d] %s ausente no SP3", i, total, prn)
            continue

        df_prn = pd.read_parquet(pq_path)

        # Skip se coordenadas já existem
        if {"x_km", "y_km", "z_km"}.issubset(df_prn.columns):
            if df_prn[["x_km", "y_km", "z_km"]].notna().all(axis=None):
                log.debug("[%d/%d] SKIP %s", i, total, pq_path.name)
                continue

        mask = (df_prn["horario"] >= ref_date) & (
            df_prn["horario"] < (ref_date + timedelta(days=1))
        )
        df_prn = df_prn[mask]
        df_prn = sp3.satellite_coordinates(df_prn, df_orb_prn, prn, ref_date)
        
        df_prn.to_parquet(pq_path, index=False)
        log.info("[%d/%d] SAT OK %s", i, total, pq_path.name)

def run(year: int, doy: int) -> None:
    """Executa o processamento para um dia específico."""
    tec_pc_folder = config.tec_pc_dir(year, doy)
    if not tec_pc_folder.exists():
        log.error("Pasta de TEC não encontrada: %s", tec_pc_folder)
        return

    try:
        df_orbit = load_or_build_orbit(year, doy)
    except Exception:
        log.exception("Erro ao carregar órbitas para %d-%03d", year, doy)
        return

    ref_date = pd.Timestamp(config.doy_to_date(year, doy))
    compute_satellite_positions(tec_pc_folder, df_orbit, ref_date)

def main():
    """Função principal para execução via linha de comando."""
    config.setup_logging()
    parser = argparse.ArgumentParser(description="Processamento de Efemérides GNSS")
    parser.add_argument("--year", type=int, required=True, help="Ano do processamento")
    parser.add_argument("--doy", type=int, required=True, help="Dia do ano (DOY)")
    args = parser.parse_args()
    run(args.year, args.doy)

def main_spyder():
    """Função para depuração facilitada no Spyder ou VS Code."""
    config.setup_logging()
    # --- Ajuste os valores abaixo para o teste que deseja realizar ---
    debug_year = 2024
    debug_doy = 264
    log.info("Iniciando depuração (main_spyder) para %d-%03d", debug_year, debug_doy)
    run(debug_year, debug_doy)

if __name__ == "__main__":
    # main()
    main_spyder()