# -*- coding: utf-8 -*-
"""Etapa 3.2 — Cálculo de TEC (pcarrier) por PRN + metadados de estações.

Para cada pickle gerado em 3.1, calcula o TEC ``pcarrier`` e escreve um
Parquet **por PRN** em ``tec_pc_dir`` (formato: uma coluna ``horario`` +
1 coluna por estação). Também gera/atualiza o ``SP_{year}-{doy}.parquet``
com os metadados (xyz e φλh) de todas as estações do dia.

Pickles que falham são movidos para ``tec_error_dir``. Estações que já
aparecem como coluna em algum parquet por PRN do mesmo dia são puladas.

Uso::

    python 03b_compute_tec.py                       # toda a árvore
    python 03b_compute_tec.py --year 2025           # ano inteiro
    python 03b_compute_tec.py --year 2025 --doy 298 # um único dia
"""

from __future__ import annotations

import logging
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from shared.config import tec_dir, tec_pc_dir, tec_error_dir, station_positions_dir, setup_logging
from shared.sp3 import times
from shared.tec import tec_rinex2

from shared.pipeline_utils import build_arg_parser, resolve_pairs
from shared.prn_store import get_processed_stations, list_files, save_prn_incremental

log = logging.getLogger("process.compute_tec")


# --------------------------------------------------------------------------
# Núcleo
# --------------------------------------------------------------------------

def process_tec(
    tec_folder: Path,
    tec_pc_folder: Path,
    tec_error_folder: Path,
    station_positions_file: Path,
    year: int,
    doy: int,
) -> None:
    tec_error_folder.mkdir(parents=True, exist_ok=True)
    station_positions_file.parent.mkdir(parents=True, exist_ok=True)

    files = list_files(tec_folder, ".pkl")
    total = len(files)
    stations_meta: list[list] = []

    # Conjunto de estações já presentes como coluna em algum parquet por PRN
    processed_stations = get_processed_stations(tec_pc_folder)
    if processed_stations:
        log.info(
            "%d estações já têm coluna nos parquets por PRN — serão puladas",
            len(processed_stations),
        )

    skipped = 0
    for i, pkl_path in enumerate(files, start=1):
        with open(pkl_path, "rb") as f:
            dic = pickle.load(f)

        head = dic.pop("header", {})
        station_name = head.get("station_name")
        xyz = head.get("xyz_geocentric", (np.nan, np.nan, np.nan))
        phi_lam_h = head.get("φλh_geodetic", (np.nan, np.nan, np.nan))

        # Metadados sempre coletados (barato) para refrescar o SP ao final
        stations_meta.append([
            station_name,
            xyz[0], xyz[1], xyz[2],
            phi_lam_h[0], phi_lam_h[1], phi_lam_h[2] / 1000.0,
        ])

        # Skip: estação já é coluna em algum parquet deste DOY
        if station_name in processed_stations:
            skipped += 1
            log.debug(
                "[%d/%d] SKIP %s (%s já é coluna)",
                i, total, pkl_path.name, station_name,
            )
            continue

        try:
            # timestamps "julianos" + cálculo de TEC pcarrier
            for key in list(dic.keys()):
                dic[key] = times(dic[key])
                dic[key] = tec_rinex2(dic[key], key, "pcarrier")

            # salva Parquet por PRN (insere nova coluna com o nome da estação)
            for key, df in dic.items():
                for prn in df["prn"].unique():
                    log.debug("Processing %s of %s", prn, pkl_path.name)
                    save_prn_incremental(
                        tec_pc_folder, prn, df, station_name, year, doy,
                    )

            log.info("[%d/%d] TEC OK  %s", i, total, pkl_path.name)

        except Exception:
            shutil.move(str(pkl_path), tec_error_folder / pkl_path.name)
            log.exception(
                "[%d/%d] TEC FAIL %s (movido para tec_error)",
                i, total, pkl_path.name,
            )

    if skipped:
        log.info(
            "  → %d de %d estações já estavam processadas (puladas)",
            skipped, total,
        )

    # --- SP file (metadados das estações) -----------------------------------
    # Sempre une com o que já existe em disco para não perder estações
    # previamente gravadas (em especial quando todas foram puladas).
    new_sp_df = pd.DataFrame(
        stations_meta,
        columns=["name", "x_km", "y_km", "z_km", "ln_deg", "lt_deg", "h_km"],
    )

    if station_positions_file.exists():
        try:
            existing = pd.read_parquet(station_positions_file)
            combined = pd.concat([existing, new_sp_df], ignore_index=True)
            df_sp = (
                combined.drop_duplicates(subset="name", keep="last")
                .sort_values("name")
                .reset_index(drop=True)
            )
        except Exception as exc:
            log.warning(
                "Erro lendo SP existente (%s), reescrevendo do zero: %s",
                station_positions_file.name, exc,
            )
            df_sp = new_sp_df.sort_values("name").reset_index(drop=True)
    else:
        df_sp = new_sp_df.sort_values("name").reset_index(drop=True)

    df_sp.to_parquet(station_positions_file, index=False)
    log.info(
        "Metadados gravados em %s (%d estações)",
        station_positions_file, len(df_sp),
    )


def run(year: int, doy: int) -> None:
    tec_folder = tec_dir(year, doy)
    tec_pc_folder = tec_pc_dir(year, doy)
    tec_error_folder = tec_error_dir(year, doy)
    sp_file = station_positions_dir() / f"SP_{year}-{doy:03d}.parquet"

    lg = logging.LoggerAdapter(log, {"year": year, "doy": doy})

    if not tec_folder.exists():
        lg.warning("Pasta de pickles inexistente: %s (rode 03a antes)", tec_folder)
        return

    lg.info("=== 3.2 TEC (pcarrier) + metadados ===")
    process_tec(tec_folder, tec_pc_folder, tec_error_folder, sp_file, year, doy)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    args = build_arg_parser(
        "Etapa 3.2 — Cálculo de TEC (pcarrier) por PRN + metadados"
    ).parse_args()

    pairs = resolve_pairs(args)
    log.info("Encontradas %d pastas year/doy para processar", len(pairs))
    for year, doy in pairs:
        log.info("=== %d / doy %03d ===", year, doy)
        run(year, doy)
