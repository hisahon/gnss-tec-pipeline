# -*- coding: utf-8 -*-
#==============================================================================
#--- @author: Honda, R. H., on: Fri Feb 24 14:03:55 2023
#==============================================================================

"""Stage 3.c — Download SP3 + consolidate + project satellite positions.

For each (year, doy):
  1) Download "Final" SP3 from CDDIS (priority: COD -> WUM -> SHA -> IAC).
  2) Concat days (-1, 0, +1, +2) into `{year}{doy}_complete.pkl` for
     edge-safe orbit interpolation downstream.
  3) For each per-PRN parquet of the day, interpolate satellite
     positions (x_km, y_km, z_km) from the orbit pickle and overwrite.

Phases 1 and 2 are idempotent. Phase 3 always recomputes (overwrites).
Requires EARTHDATA_USERNAME / EARTHDATA_PASSWORD.

Usage:
    python 03c_download_sp3.py
    python 03c_download_sp3.py --year 2025
    python 03c_download_sp3.py --year 2025 --doy 298
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
import os

import pandas as pd

from shared import sp3
from shared.config import (
    ephemeris_dir,
    ephemeris_byprn_dir,
    doy_to_date,
    setup_logging,
    tec_pc_dir,  # adjust the import name if your codebase uses a different one
)
from shared.earthdata_download import download_precise_orbits
from shared.pipeline_utils import build_arg_parser, resolve_pairs

from shared.prn_store import load_tec_with_neighbors

log = logging.getLogger("process.download_sp3")

# `date` (not `datetime`) so subtraction works with either type.
GPS_EPOCH = date(1980, 1, 6)


def calculate_gnss_week(dt) -> int:
    """Full weeks elapsed since the GPS epoch (1980-01-06)."""
    d = dt.date() if isinstance(dt, datetime) else dt
    return (d - GPS_EPOCH).days // 7


def prn_from_file_path(path: Path) -> str:
    """PRN id is the first underscore-separated token of the file stem."""
    # Replace with your shared helper if you already have one.
    return path.stem.split("_")[0]

# ---------------------------------------------------------------------------
# Pickle builder
# ---------------------------------------------------------------------------
 
def build_complete_pickle(year: int, doy: int, sp3_file: Path,
                          lg: logging.LoggerAdapter) -> bool:
    """Concat days (-1..+2) of SP3 into `sp3_file`. Returns True on success."""
    dt_in = doy_to_date(year, doy)
    # concat_all wants the base folder, not the day folder.
    eph_folder = sp3_file.parent.parent.parent
    lg.info("GNSS week %d — building %s",
            calculate_gnss_week(dt_in), sp3_file.name)
 
    try:
        df_orbit = sp3.concat_all(str(eph_folder), dt_in)
    except Exception:
        lg.exception("SP3 consolidation failed")
        return False
 
    # Don't write an empty pickle — would poison future skip checks.
    if df_orbit.empty:
        lg.warning("Empty orbit DataFrame — pickle not written")
        return False
 
    df_orbit.to_pickle(sp3_file)
    lg.info("Pickle written: %s (%d rows)", sp3_file, len(df_orbit))
    return True



# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run(year: int, doy: int) -> None:
    lg = logging.LoggerAdapter(log, {"year": year, "doy": doy})
    eph_dir = ephemeris_dir(year, doy)
    eph_dir.mkdir(parents=True, exist_ok=True)

    sp3_file = eph_dir / f"{year}{doy:03d}_complete.pkl"

    # --- Ensure pickle exists ---
    if sp3_file.exists():
        lg.info("SKIP build — %s already exists", sp3_file.name)
    else:
        lg.info("=== 3.c SP3 download (FIN) ===")
        try:
            download_precise_orbits(doy_to_date(year, doy))
        except Exception:
            lg.exception("SP3 download failed")
            return
        if not build_complete_pickle(year, doy, sp3_file, lg):
            return  # nothing to project against

    # --- Project positions onto per-PRN parquets ---
    satellite_positions2(year, doy)


# def satellite_positions(year: int, doy: int) -> None:
#     """Add interpolated (x_km, y_km, z_km) to each per-PRN parquet of the day."""
#     lg = logging.LoggerAdapter(log, {"year": year, "doy": doy})

#     sp3_file = ephemeris_dir(year, doy) / f"{year}{doy:03d}_complete.pkl"
#     if not sp3_file.exists():
#         lg.warning("SKIP positions (orbit pickle missing)")
#         return

#     parquet_ref = tec_pc_dir(year, doy)
#     parquet_dir = parquet_ref / "E"
    
    
#     """
#     ##############################################################
#     #############################################################
#     ##############################################################
#     #############################################################
    
    
#     Aqui eu preciso fazer a rotina ler e inserir as posições dos satelites 
#     Em cada uma das estações, talvez seja interessante salvar esses arquivos separados
#     um para cada satelite, minha ultima ideia,
    
#     salvar um {year}{doy}_E0X.parquet.
    
#     ^ ^ ^
#     Pensa nisso
    
    
#     #########################################################
#     #########################################################
#     ########################################################
#     """
    
    
    
#     #load_tec_with_neighbors(year, doy, with_neighbors=False)
#     files = sorted(parquet_dir.glob("*.parquet"))
#     if not files:
#         lg.warning("SKIP positions (no parquets in %s)", parquet_dir)
#         return

#     df_orbit = pd.read_pickle(sp3_file)
#     dt_in = pd.Timestamp(doy_to_date(year, doy))
#     # ref_date = start of orbit window (-1d) so seconds stay non-negative.
#     ref_date = dt_in - pd.Timedelta(days=1)
#     df_orbit["sec"] = (
#         (df_orbit["horario"] - ref_date).dt.total_seconds().astype(int).to_numpy()
#     )

#     for parquet in files:
#         prn = prn_from_file_path(parquet)

#         # Drop dup timestamps: interpolators reject non-strictly-increasing x.
#         df_orb_prn = (
#             df_orbit[df_orbit["prn"] == prn]
#             .sort_values("sec")
#             .drop_duplicates(subset=["sec"], keep="first")
#         )
#         if df_orb_prn.empty:
#             lg.info("%s missing in orbit file", prn)
#             continue

#         df_prn = pd.read_parquet(parquet)
#         # Keep only the target day; orbit pickle covers a wider window.
#         df_prn = df_prn[
#             (df_prn["horario"] >= dt_in)
#             & (df_prn["horario"] < dt_in + pd.Timedelta(days=1))
#         ]

#         try:
#             df_prn = sp3.satellite_coordinates(df_prn, df_orb_prn, prn, ref_date)
#             df_prn.to_parquet(parquet, index=False)
#             lg.info("%s positions saved", parquet.name)
#         except Exception:
#             lg.exception("%s position calc failed", prn)



def satellite_positions2(year: int, doy: int) -> None:
    """Add interpolated (x_km, y_km, z_km) to each per-PRN parquet of the day."""
    lg = logging.LoggerAdapter(log, {"year": year, "doy": doy})

    sp3_file = ephemeris_dir(year, doy) / f"{year}{doy:03d}_complete.pkl"
    if not sp3_file.exists():
        lg.warning("SKIP positions (orbit pickle missing)")
        return

    df_orbit = pd.read_pickle(sp3_file)
    dt_in = pd.Timestamp(doy_to_date(year, doy))
    
    # ref_date = start of orbit window (-1d) so seconds stay non-negative.
    ref_date = dt_in - pd.Timedelta(days=1)
    df_orbit["sec"] = (
        (df_orbit["horario"] - ref_date).dt.total_seconds().astype(int).to_numpy()
    )
    
    # =========================================================
    # NOVA ROTINA INTEGRADA: Separando df_orbit por PRN
    # =========================================================
    
    # 1. Define a pasta onde os arquivos separados serão salvos
    # (Estou criando uma subpasta "orbits_por_prn" dentro do diretório do dia)
    pasta_destino = ephemeris_byprn_dir(year, doy)
    pasta_destino.mkdir(parents=True, exist_ok=True) # pathlib substitui o os.makedirs

    lg.info(f"Separando {len(df_orbit)} registros de órbita por PRN...")

    # 2. Agrupa pelo PRN e salva cada grupo
    # ATENÇÃO: Verifique se o nome da coluna no seu pickle é 'prn' mesmo (tudo minúsculo)
    for prn_valor, df_grupo in df_orbit.groupby('prn'):
        
        prn_limpo = str(prn_valor).strip()
        
        # Como o código comentado usa parquet, recomendo manter o formato
        nome_arquivo = f"orbitprn_{year}{doy:03d}_{prn_limpo}.parquet"
        caminho_completo = pasta_destino / nome_arquivo
        
        # Salva em parquet. Se realmente quiser CSV, troque para:
        # df_grupo.to_csv(caminho_completo, index=False)
        
        if caminho_completo.exists():
            # print(f"{caminho_completo.name} ja existe")
            continue
        
        df_grupo[['horario','x_km','y_km','z_km']].to_parquet(caminho_completo, index=False)
        
    lg.info(f"Arquivos separados e salvos em: {pasta_destino}")
    ...

    # for parquet in files:
    #     prn = prn_from_file_path(parquet)

    #     # Drop dup timestamps: interpolators reject non-strictly-increasing x.
    #     df_orb_prn = (
    #         df_orbit[df_orbit["prn"] == prn]
    #         .sort_values("sec")
    #         .drop_duplicates(subset=["sec"], keep="first")
    #     )
    #     if df_orb_prn.empty:
    #         lg.info("%s missing in orbit file", prn)
    #         continue

    #     df_prn = pd.read_parquet(parquet)
    #     # Keep only the target day; orbit pickle covers a wider window.
    #     df_prn = df_prn[
    #         (df_prn["horario"] >= dt_in)
    #         & (df_prn["horario"] < dt_in + pd.Timedelta(days=1))
    #     ]

    #     try:
    #         df_prn = sp3.satellite_coordinates(df_prn, df_orb_prn, prn, ref_date)
    #         df_prn.to_parquet(parquet, index=False)
    #         lg.info("%s positions saved", parquet.name)
    #     except Exception:
    #         lg.exception("%s position calc failed", prn)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    args = build_arg_parser(
        "Stage 3.c — Download SP3 from CDDIS + consolidate + project positions"
    ).parse_args()

    pairs = resolve_pairs(args, referencefolder="tecRAWReferenceFolder")
    log.info("Found %d year/doy folders to process", len(pairs))
    # pairs = [(2025,317)]
    for year, doy in pairs:
        log.info("=== %d / doy %03d ===", year, doy)
        run(year, doy)