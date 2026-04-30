# -*- coding: utf-8 -*-
"""Etapa 3.2 — Cálculo de TEC (pcarrier) por PRN + metadados de estações.
[OTIMIZADO COM DUCKDB]

Diferenças vs. versão original
------------------------------
- O cálculo do TEC continua igual (``times`` + ``tec_rinex2`` por chave do
  pickle), mas em vez de chamar ``save_prn_incremental`` dentro do loop
  ``por estação × por PRN``, o resultado de cada estação é *empilhado* em
  formato long (``prn, horario, station, tec``) num **staging in-memory
  do DuckDB**.
- Ao final, para cada PRN distinto presente no staging, é feito **um
  único** ``PIVOT`` dinâmico + ``FULL OUTER JOIN`` com o parquet
  existente (se houver) e **uma única escrita** do parquet daquele PRN.
- I/O de parquet cai de ``O(N_estacoes × N_PRNs)`` para ``O(N_PRNs)``.
- O ``SP_{year}-{doy}.parquet`` (metadados das estações) também é
  consolidado via DuckDB (``UNION`` mantendo a versão "nova").

Pickles que falham continuam sendo movidos para ``tec_error_dir``.
Estações que já aparecem como coluna em algum parquet por PRN do mesmo
dia continuam sendo puladas (via ``get_processed_stations``).

Requer: ``pip install duckdb``

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

import duckdb
import numpy as np
import pandas as pd

from shared.config import (
    tec_dir, tec_pc_dir, tec_error_dir, station_positions_dir, setup_logging,
)
from shared.sp3 import times
from shared.tec import tec_rinex2

from shared.pipeline_utils import build_arg_parser, resolve_pairs
from shared.prn_store import get_processed_stations, list_files

log = logging.getLogger("process.compute_tec")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _build_long_frame(dic: dict, station_name: str) -> pd.DataFrame:
    """Achata o dicionário de DataFrames (saída de ``tec_rinex2``) em um
    único frame long-format ``(prn, horario, station, tec)`` pronto para
    fazer ``append`` no staging do DuckDB."""
    parts = []
    for _, df in dic.items():
        if df is None or len(df) == 0:
            continue
        sub = df[["prn", "horario", "tec_pcarrier"]].copy()
        sub["station"] = station_name
        # sub = sub.rename(columns={"pcarrier": "tec"})
        parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=["prn", "horario", "station", "tec_pcarrier"])
    return pd.concat(parts, ignore_index=True)


def _consolidate_sp(
    new_sp_df: pd.DataFrame,
    sp_file: Path,
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Une os metadados novos com o parquet SP já gravado, mantendo a
    versão "nova" (equivalente a ``drop_duplicates(keep='last')`` da
    versão original)."""
    if not sp_file.exists():
        return new_sp_df.sort_values("name").reset_index(drop=True)

    try:
        con.register("_new_sp", new_sp_df)
        df_sp = con.execute(f"""
            SELECT * FROM _new_sp
            UNION ALL
            SELECT * FROM read_parquet('{sp_file.as_posix()}')
            WHERE name NOT IN (SELECT name FROM _new_sp)
            ORDER BY name
        """).df()
        con.unregister("_new_sp")
        return df_sp
    except Exception as exc:
        log.warning(
            "Erro consolidando SP existente (%s), reescrevendo do zero: %s",
            sp_file.name, exc,
        )
        return new_sp_df.sort_values("name").reset_index(drop=True)


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
    tec_pc_folder.mkdir(parents=True, exist_ok=True)
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

    # DuckDB em memória; multi-thread por padrão. Para datasets muito
    # grandes pode-se trocar por arquivo: duckdb.connect("/tmp/stage.duckdb")
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE staging (
            prn               VARCHAR,
            horario           TIMESTAMP,
            tec_pcarrier      DOUBLE,
            station           VARCHAR
        )
    """)

    skipped = 0
    staged = 0  # nº de estações efetivamente empilhadas no staging
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
            if len(dic.keys()) == 0:
                continue
            
            for key in list(dic.keys()):                
                dic[key] = times(dic[key])
                dic[key] = tec_rinex2(dic[key], key, "pcarrier")

            long_df = _build_long_frame(dic, station_name)
            if not long_df.empty:
                con.append("staging", long_df)
                staged += 1

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

    # ----------------------------------------------------------------------
    # PIVOT + merge por PRN — exatamente UMA escrita de parquet por PRN
    # ----------------------------------------------------------------------
    if staged > 0:
        prns = [r[0] for r in con.execute(
            "SELECT DISTINCT prn FROM staging ORDER BY prn"
        ).fetchall()]
        log.info(
            "Consolidando %d PRNs via DuckDB (1 escrita parquet por PRN)",
            len(prns),
        )

        for prn in prns:
            prn_file = tec_pc_folder / f"TECPC_{year}-{doy:03d}_{prn}.parquet"
            tmp_file = prn_file.with_suffix(".parquet.tmp")

            # PIVOT dinâmico: 1 coluna por estação que entrou nesta execução.
            # Usa first(tec) porque para um (horario, station, prn) só há
            # um valor de TEC; equivalente a max/avg para esse caso.
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE _new_pivot AS
                PIVOT (
                    SELECT horario, station, tec_pcarrier
                    FROM staging
                    WHERE prn = '{prn}'
                ) ON station USING first(tec_pcarrier) GROUP BY horario
            """)

            if prn_file.exists():
                # Junta colunas novas + colunas antigas numa única passada.
                # FULL OUTER JOIN USING(horario) preserva o eixo temporal
                # completo de ambos os lados (importante quando estações
                # têm janelas de observação distintas).
                con.execute(f"""
                    COPY (
                        SELECT *
                        FROM read_parquet('{prn_file.as_posix()}') AS old
                        FULL OUTER JOIN _new_pivot USING (horario)
                        ORDER BY horario
                    ) TO '{tmp_file.as_posix()}' (FORMAT PARQUET)
                """)
            else:
                con.execute(f"""
                    COPY (
                        SELECT * FROM _new_pivot ORDER BY horario
                    ) TO '{tmp_file.as_posix()}' (FORMAT PARQUET)
                """)

            # Substituição atômica para nunca deixar o parquet meio-escrito
            tmp_file.replace(prn_file)
            log.debug("PRN %s gravado (%s)", prn, prn_file.name)
    else:
        log.info("Nada novo para gravar (todas as estações já processadas).")

    # ----------------------------------------------------------------------
    # SP file (metadados das estações) — consolidado também via DuckDB
    # ----------------------------------------------------------------------
    new_sp_df = pd.DataFrame(
        stations_meta,
        columns=["name", "x_km", "y_km", "z_km", "ln_deg", "lt_deg", "h_km"],
    )
    df_sp = _consolidate_sp(new_sp_df, station_positions_file, con)
    df_sp.to_parquet(station_positions_file, index=False)
    log.info(
        "Metadados gravados em %s (%d estações)",
        station_positions_file, len(df_sp),
    )

    con.close()


def run(year: int, doy: int) -> None:
    tec_folder = tec_dir(year, doy)
    tec_pc_folder = tec_pc_dir(year, doy)
    tec_error_folder = tec_error_dir(year, doy)
    sp_file = station_positions_dir() / f"SP_{year}-{doy:03d}.parquet"

    lg = logging.LoggerAdapter(log, {"year": year, "doy": doy})

    if not tec_folder.exists():
        lg.warning("Pasta de pickles inexistente: %s (rode 03a antes)", tec_folder)
        return

    lg.info("=== 3.2 TEC (pcarrier) + metadados [DuckDB] ===")
    process_tec(tec_folder, tec_pc_folder, tec_error_folder, sp_file, year, doy)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    args = build_arg_parser(
        "Etapa 3.2 — Cálculo de TEC (pcarrier) por PRN + metadados [DuckDB]"
    ).parse_args()

    pairs = resolve_pairs(args, 'rinexPKLReferenceFolder')
    log.info("Encontradas %d pastas year/doy para processar", len(pairs))
    
    # pairs = [(2025,316)]
    for year, doy in pairs:
        log.info("=== %d / doy %03d ===", year, doy)
        run(year, doy)
