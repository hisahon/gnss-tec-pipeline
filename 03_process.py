# -*- coding: utf-8 -*-
"""Etapa 03 — Cálculo de TEC, metadados de estações e posições de satélite.

Fluxo:

1. ``process_rinex_to_pickle`` — lê todo ``.??o`` em ``rinex_dir`` e salva
   um ``.pkl`` correspondente em ``tec_dir``.
2. ``process_tec``             — para cada pickle, calcula o TEC
   ``pcarrier`` e escreve um Parquet **por PRN** em ``tec_pc_dir``
   (formato: uma coluna ``horario`` + 1 coluna por estação). Também gera
   ``SP_{year}-{doy}.parquet`` com metadados de todas as estações.
3. ``compute_satellite_positions`` — interpola as coordenadas do satélite
   a partir do SP3 consolidado e as adiciona aos Parquets por PRN.

Uso::

    python 03_process.py --year 2025 --doy 298
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config

# Módulos internos do pipeline
import rinex as rnx
import sp3
import tec as tc

log = logging.getLogger("process")

PRN_RE = re.compile(r"([GRECIJS]\d{2})$")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def list_files(folder: Path, extension: str) -> list[Path]:
    """Retorna recursivamente todos os arquivos com a extensão dada."""
    return sorted(folder.rglob(f"*{extension}"))


def prn_from_path(path: Path) -> str:
    """Extrai o identificador de PRN (ex.: ``G12``) do nome do arquivo."""
    stem = path.stem
    tok = stem.split("_")[-1]
    m = PRN_RE.search(tok) or PRN_RE.search(stem)
    return m.group(1) if m else tok


def _first_valid(s: pd.Series):
    """Agg helper: primeiro valor não-nulo da série."""
    return s.dropna().iloc[0] if s.notna().any() else pd.NA


def save_prn_incremental(
    out_dir: Path,
    prn: str,
    df: pd.DataFrame,
    station_name: str,
    year: int,
    doy: int,
) -> Path:
    """Mantém um Parquet por PRN com colunas ``horario`` + uma por estação.

    Incrementa linhas e colunas conforme novas estações chegam, fazendo
    ``coalesce`` por horário (mantém o primeiro valor não-nulo).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"TECPC_{year}-{doy:03d}_{prn}.parquet"

    df_prn = df[df["prn"] == prn][["horario", "tec_pcarrier"]].copy()
    if df_prn["tec_pcarrier"].isna().all():
        return fpath

    df_new = (
        df_prn.rename(columns={"tec_pcarrier": station_name})
        .assign(horario=lambda d: pd.to_datetime(d["horario"]))
        .drop_duplicates(subset=["horario"], keep="last")
        .sort_values("horario")
    )

    if fpath.exists():
        df_disk = pd.read_parquet(fpath)
        df_disk["horario"] = pd.to_datetime(df_disk["horario"], errors="coerce")

        if station_name in df_disk.columns:
            df_cat = pd.concat([df_disk, df_new], ignore_index=True, sort=False)
            agg = {c: _first_valid for c in df_cat.columns if c != "horario"}
            df_merged = (
                df_cat.groupby("horario", as_index=False)
                .agg(agg)
                .sort_values("horario")
            )
        else:
            df_merged = df_disk.merge(df_new, on="horario", how="outer", sort=True)
    else:
        df_merged = df_new

    cols = ["horario"] + [c for c in df_merged.columns if c != "horario"]
    df_merged[cols].to_parquet(fpath, index=False)
    return fpath


# --------------------------------------------------------------------------
# Etapa 3.1 — RINEX → pickle
# --------------------------------------------------------------------------

def process_rinex_to_pickle(rinex_folder: Path, tec_folder: Path) -> None:
    tec_folder.mkdir(parents=True, exist_ok=True)
    files = list_files(rinex_folder, "o")
    total = len(files)

    for i, rinex_path in enumerate(files, start=1):
        out_path = tec_folder / f"{rinex_path.stem}.pkl"
        if out_path.exists():
            log.info("[%d/%d] SKIP %s (pickle já existe)", i, total, rinex_path.name)
            continue

        try:
            dic = rnx.read_rinex(str(rinex_path))
            with open(out_path, "wb") as f:
                pickle.dump(dic, f)
            log.info("[%d/%d] OK   %s", i, total, rinex_path.name)
        except Exception:
            log.exception("[%d/%d] FAIL %s", i, total, rinex_path.name)


# --------------------------------------------------------------------------
# Etapa 3.2 — TEC (pcarrier) + metadados
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

    for i, pkl_path in enumerate(files, start=1):
        with open(pkl_path, "rb") as f:
            dic = pickle.load(f)

        head = dic.pop("header", {})
        xyz = head.get("xyz_geocentric", (np.nan, np.nan, np.nan))
        phi_lam_h = head.get("φλh_geodetic", (np.nan, np.nan, np.nan))
        stations_meta.append([
            head.get("station_name"),
            xyz[0], xyz[1], xyz[2],
            phi_lam_h[0], phi_lam_h[1], phi_lam_h[2] / 1000.0,
        ])

        try:
            # timestamps "julianos" + cálculo de TEC pcarrier
            for key in list(dic.keys()):
                dic[key] = sp3.times(dic[key])
                dic[key] = tc.tec_rinex2(dic[key], key, "pcarrier")

            # salva Parquet por PRN
            for key, df in dic.items():
                for prn in df["prn"].unique():
                    log.debug("Processing %s of %s", prn, pkl_path.name)
                    save_prn_incremental(
                        tec_pc_folder, prn, df, head["station_name"], year, doy,
                    )

            log.info("[%d/%d] TEC OK  %s", i, total, pkl_path.name)

        except Exception:
            shutil.move(str(pkl_path), tec_error_folder / pkl_path.name)
            log.exception("[%d/%d] TEC FAIL %s (movido para tec_error)", i, total, pkl_path.name)

    df_sp = pd.DataFrame(
        stations_meta,
        columns=["name", "x_km", "y_km", "z_km", "ln_deg", "lt_deg", "h_km"],
    ).sort_values("name").reset_index(drop=True)
    df_sp.to_parquet(station_positions_file, index=False)
    log.info("Metadados de estações gravados em %s", station_positions_file)


# --------------------------------------------------------------------------
# Etapa 3.3 — Posições dos satélites
# --------------------------------------------------------------------------

def compute_satellite_positions(
    tec_pc_folder: Path,
    df_orbit: pd.DataFrame,
    ref_date: pd.Timestamp,
) -> None:
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
        mask = (df_prn["horario"] >= ref_date) & (
            df_prn["horario"] < (ref_date + timedelta(days=1))
        )
        df_prn = df_prn[mask]

        df_prn = sp3.satellite_coordinates(df_prn, df_orb_prn, prn, ref_date)
        df_prn.to_parquet(pq_path, index=False)
        log.info("[%d/%d] SAT OK %s", i, total, pq_path.name)


# --------------------------------------------------------------------------
# Efemérides (SP3 consolidado)
# --------------------------------------------------------------------------

def load_or_build_orbit(year: int, doy: int) -> pd.DataFrame:
    eph_dir = config.ephemeris_dir(year, doy)
    eph_dir.mkdir(parents=True, exist_ok=True)

    sp3_pickle = eph_dir / f"{year}{doy:03d}_complete.pkl"
    if sp3_pickle.exists():
        log.info("Lendo SP3 consolidado de %s", sp3_pickle)
        return pd.read_pickle(sp3_pickle)

    dt_in = config.doy_to_date(year, doy)
    week = config.gnss_week(dt_in)
    log.info("SP3 não encontrado. Construindo (GNSS week %d)...", week)

    # Caminho histórico: concat_all recebe a pasta raiz de efemérides
    df_orbit = sp3.concat_all(str(config.DB_ROOT / "gnss" / "ephemeris"), dt_in)
    df_orbit.to_pickle(sp3_pickle)
    return df_orbit


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------

def run(year: int, doy: int) -> None:
    rinex_folder = config.rinex_dir(year, doy)
    tec_folder = config.tec_dir(year, doy)
    tec_pc_folder = config.tec_pc_dir(year, doy)
    tec_error_folder = config.tec_error_dir(year, doy)
    sp_file = config.station_positions_dir() / f"SP_{year}-{doy:03d}.parquet"

    log.info("=== 3.1 RINEX → pickle ===")
    process_rinex_to_pickle(rinex_folder, tec_folder)

    log.info("=== 3.2 TEC (pcarrier) + metadados ===")
    process_tec(tec_folder, tec_pc_folder, tec_error_folder, sp_file, year, doy)

    log.info("=== 3.3 Posições dos satélites ===")
    df_orbit = load_or_build_orbit(year, doy)
    ref_date = pd.Timestamp(config.doy_to_date(year, doy))
    compute_satellite_positions(tec_pc_folder, df_orbit, ref_date)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cálculo de TEC + posições de satélite")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--doy", type=int, required=True)
    return p.parse_args()


if __name__ == "__main__":
    config.setup_logging()
    args = parse_args()
    run(args.year, args.doy)
