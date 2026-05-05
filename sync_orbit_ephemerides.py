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
    satellite_positions(year, doy)



def satellite_positions(year: int, doy: int) -> None:
    """Split the day's orbit pickle into one parquet per PRN."""
    lg = logging.LoggerAdapter(log, {"year": year, "doy": doy})

    sp3_file = ephemeris_dir(year, doy) / f"{year}{doy:03d}_complete.pkl"
    if not sp3_file.exists():
        lg.warning("SKIP positions (orbit pickle missing)")
        return

    df_orbit = pd.read_pickle(sp3_file)
    dt_in = pd.Timestamp(doy_to_date(year, doy))
    ref_date = dt_in - pd.Timedelta(days=1)  # keeps `sec` non-negative

    out_dir = ephemeris_byprn_dir(year, doy)
    out_dir.mkdir(parents=True, exist_ok=True)

    lg.info("Splitting orbit pickle into per-PRN parquets (%d PRNs, %d rows)",
            df_orbit["prn"].nunique(), len(df_orbit))

    written = skipped = failed = 0
    for prn_value, df_group in df_orbit.groupby("prn"):
        # SP3 parsers sometimes leave whitespace on PRN tokens.
        prn = str(prn_value).strip()
        out_file = out_dir / f"orbitprn_{year}{doy:03d}_{prn}.parquet"

        if out_file.exists():
            skipped += 1
            continue

        try:
            df_group[["horario", "x_km", "y_km", "z_km"]].to_parquet(
                out_file, index=False
            )
            written += 1
        except Exception:
            lg.exception("PRN %s failed to write", prn)
            failed += 1

    lg.info("Per-PRN orbits in %s: %d written, %d skipped, %d failed",
            out_dir, written, skipped, failed)



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