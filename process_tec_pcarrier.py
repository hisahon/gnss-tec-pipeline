# -*- coding: utf-8 -*-
"""Etapa 3.2 — Cálculo de TEC (pcarrier) por estação + metadados.
[LAYOUT LONG PARTICIONADO — escrita pura, sem read-modify-write]

Mudança vs. versão anterior
---------------------------
Antes: 1 parquet por PRN × DOY, com 1 coluna por estação (formato wide).
       Adicionar 1 estação exigia ler o parquet antigo, FULL OUTER JOIN
       e reescrever — IO de leitura O(N²) conforme estações são adicionadas.

Agora: 1 parquet por (estação × constelação × DOY), em formato long.

    tec_pc/
      {year}/{doy:03d}/
        G/ALAR.parquet    ← prn, horario, tec_pcarrier
        G/BRAZ.parquet
        R/ALAR.parquet
        ...

Adicionar uma estação é ESCRITA PURA: nada de parquets antigos é lido.
1ª e 600ª estação têm o mesmo custo de IO. NaN deixa de existir
(long só armazena observações reais).

Operações via shell continuam triviais::

    find tec_pc -name "ALAR.parquet" -delete       # apagar estação
    find tec_pc -path "*/C/*" -delete              # apagar BeiDou
    rm -rf tec_pc/2025/298/                        # apagar 1 dia
    ls tec_pc/2025/298/G/                          # estações de 1 dia/const

Para análises (média móvel cruzando dias, formato wide, etc.) ver
``tec_query.py`` — DuckDB lê os arquivos sob demanda.

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

from shared.config import (
    tec_dir, tec_pc_dir, tec_error_dir, station_positions_dir, setup_logging,
)
from shared.sp3 import times
from shared.tec import tec_rinex2

from shared.pipeline_utils import build_arg_parser, resolve_pairs
from shared.prn_store import list_files

log = logging.getLogger("process.compute_tec")

# Colunas mínimas escritas em cada parquet por estação×constelação
_TEC_COLS = ["prn", "horario", "tec_pcarrier"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _station_from_filename(pkl_path: Path) -> str:
    """Convenção RINEX/IGS: 4 primeiros caracteres do filename = station id."""
    return pkl_path.stem[:4].lower()


def _get_processed_stations(tec_pc_folder: Path) -> set[str]:
    """
    Estações já processadas neste DOY = qualquer station que tenha pelo menos
    um arquivo {const}/{station}.parquet.

    Custo: 1 glob raso. Diretórios pequenos (~600 arquivos), trivial em SSD.
    """
    if not tec_pc_folder.exists():
        return set()
    return {p.stem.split('_')[1].lower() for p in tec_pc_folder.glob("*/*.parquet")}


def _extract_station_meta(head: dict) -> tuple[str | None, list]:
    """Extrai (station_name, linha_de_metadados) do header do pickle."""
    station_name = head.get("station_name")
    xyz = head.get("xyz_geocentric", (np.nan, np.nan, np.nan))
    phi_lam_h = head.get("φλh_geodetic", (np.nan, np.nan, np.nan))
    meta_row = [
        station_name,
        xyz[0], xyz[1], xyz[2],
        phi_lam_h[0], phi_lam_h[1], phi_lam_h[2] / 1000.0,
    ]
    return station_name, meta_row


def _compute_tec_pcarrier(dic: dict) -> dict:
    """Aplica times() + tec_rinex2(..., 'pcarrier') in-place para cada PRN."""
    for key in list(dic.keys()):
        dic[key] = times(dic[key])
        dic[key] = tec_rinex2(dic[key], key, "pcarrier")
    return dic


def _group_by_constellation(dic: dict) -> dict[str, list[pd.DataFrame]]:
    """
    Agrupa os DataFrames de PRN por constelação (1ª letra do PRN: G/R/E/C).
    Devolve apenas frames não-vazios.
    """
    by_const: dict[str, list[pd.DataFrame]] = {}
    for prn, df in dic.items():
        if df is None or len(df) == 0:
            continue
        cols = [c for c in _TEC_COLS if c in df.columns]
        if len(cols) < len(_TEC_COLS):
            # Algum PRN não tem as colunas esperadas — silenciosamente ignorado
            continue
        const = prn[0]  # 'G', 'R', 'E', 'C'
        by_const.setdefault(const, []).append(df[_TEC_COLS])
    return by_const


def _write_station_parquets(
    by_const: dict[str, list[pd.DataFrame]],
    tec_pc_folder: Path,
    station: str,
) -> int:
    """
    Escreve 1 parquet por constelação dessa estação. Retorna o nº de arquivos
    escritos. Escrita pura: nenhum parquet existente é lido.

    Dtypes otimizados:
      - tec_pcarrier: float32 (precisão sobra pra TEC)
      - horario: timestamp[s] (TEC raramente tem subsegundo útil)
      - prn: dictionary (categórica) — economiza muito após zstd

    Compressão zstd nível 9 — bom equilíbrio tamanho/CPU em séries temporais.
    """
    written = 0
    for const, parts in by_const.items():
        merged = pd.concat(parts, ignore_index=True)
        if merged.empty:
            continue

        merged["tec_pcarrier"] = merged["tec_pcarrier"].astype("float32")
        merged["horario"] = pd.to_datetime(merged["horario"]).astype("datetime64[s]")
        merged["prn"] = merged["prn"].astype("category")
        merged = merged.sort_values(["prn", "horario"], kind="stable")

        out_dir = tec_pc_folder / const
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{year}{doy:03d}{const}_{station}.parquet"
        tmp_path = out_path.with_suffix(".parquet.tmp")

        merged.to_parquet(
            tmp_path,
            compression="zstd",
            compression_level=9,
            index=False,
        )
        tmp_path.replace(out_path)  # rename atômico
        written += 1
    return written


def _write_station_parquet_merged(
    by_const: dict[str, list[pd.DataFrame]],
    tec_pc_folder: Path,
    station: str,
    year: int,
    doy: int,
                            ) -> int:
    """
    Write a single parquet per station containing all constellations stacked.
    """
    # Stack all dataframes, tagging each with its constellation
    all_parts = []
    for const, parts in by_const.items():
        for df in parts:
            df = df.copy()
            # df["constellation"] = const
            all_parts.append(df)

    # Nothing to save
    if not all_parts:
        return 0

    # Concatenate all dataframes into one
    merged = pd.concat(all_parts, ignore_index=True)

    # Optimize dtypes
    merged["tec_pcarrier"] = merged["tec_pcarrier"].astype("float32")
    merged["horario"] = pd.to_datetime(merged["horario"]).astype("datetime64[s]")
    merged["prn"] = merged["prn"].astype("category")
    # merged["constellation"] = merged["constellation"].astype("category")

    # Stable sort by constellation, prn, time
    merged = merged.sort_values(
        ["prn", "horario"],
        kind="stable",
        ignore_index=True,
    )

    # Build output paths
    tec_pc_folder.mkdir(parents=True, exist_ok=True)
    out_path = tec_pc_folder / f"{station}_{year}{doy:03d}.parquet"
    tmp_path = out_path.with_suffix(".parquet.tmp")

    # Write to a temp file then atomically rename
    merged.to_parquet(
        tmp_path,
        compression="zstd",
        compression_level=9,
        index=False,
    )
    tmp_path.replace(out_path)

    return 1


def _consolidate_sp(
    new_sp_df: pd.DataFrame,
    sp_file: Path,
) -> pd.DataFrame:
    """
    Consolida metadados das estações (SP file). Mantém versão 'nova' quando
    há colisão de nome. Sem DuckDB: um pd.concat + drop_duplicates resolve.
    """
    if not sp_file.exists():
        return new_sp_df.sort_values("name").reset_index(drop=True)

    try:
        old = pd.read_parquet(sp_file)
        df = pd.concat([new_sp_df, old], ignore_index=True)
        df = df.drop_duplicates(subset=["name"], keep="first")
        return df.sort_values("name").reset_index(drop=True)
    except Exception as exc:
        log.warning(
            "Erro consolidando SP existente (%s), reescrevendo do zero: %s",
            sp_file.name, exc,
        )
        return new_sp_df.sort_values("name").reset_index(drop=True)


# --------------------------------------------------------------------------
# Núcleo
# --------------------------------------------------------------------------


def _discard_stations_without_positions(meta_row):
    
    sanity_radius_km = 1000.0
                            
    x_km = float(meta_row[1])
    y_km = float(meta_row[2])
    z_km = float(meta_row[3])
        
    # Calcula a distância do centro da Terra até a estação
    center_displacement = np.sqrt(x_km**2 + y_km**2 + z_km**2)
    
    # O "Sanity Check": se estiver dentro da Terra, pule!
    if center_displacement < sanity_radius_km:
        # print(f"Estação {colunas[0]} descartada. Distância: {distancia_ao_centro:.2f} km")
        return True
    else:
        return False
        
        
def process_tec(
    tec_folder: Path,
    tec_pc_folder: Path,
    tec_error_folder: Path,
    station_positions_file: Path,
    year: int,
    doy: int,
) -> None:
    # ---- Setup de pastas -------------------------------------------------
    for folder in (tec_pc_folder, tec_error_folder, station_positions_file.parent):
        folder.mkdir(parents=True, exist_ok=True)

    # ---- Inventário de entrada e estado anterior ------------------------
    files = list_files(tec_folder, ".pkl")
    total = len(files)
    stations_meta: list[list] = []

    processed_stations = _get_processed_stations(tec_pc_folder)
    if processed_stations:
        log.info(
            "%d estações já têm arquivos no DOY (%d) — serão puladas",
            len(processed_stations), doy,
        )

    # ---- Loop principal: 1 estação = 1 leitura de pickle + N escritas ---
    skipped_already = 0   # já processada (parquet existe)
    skipped_no_pos  = 0   # sem coordenadas no header
    skipped_empty   = 0   # pickle vazio ou sem dado útil
    written_files = 0
    written_stations = 0

    for i, pkl_path in enumerate(files, start=1):
        # 1) Skip barato pelo nome do arquivo, antes de qualquer IO pesado
        station_from_name = _station_from_filename(pkl_path)
        if station_from_name in processed_stations:
            skipped_already += 1
            log.debug("[%d/%d] SKIP %s (já processada)", i, total, pkl_path.name)
            continue

        try:
            
            # 0) Skip canônico pelo path real de saída (case-correct, antes do trabalho pesado)
            out_path = tec_pc_folder / f"{pkl_path.stem[0:4]}_{year}{doy:03d}.parquet"
            if out_path.exists():
                skipped_already += 1
                log.debug("[%d/%d] SKIP %s (parquet já existe)", i, total, pkl_path.name)
                continue
            
            with open(pkl_path, "rb") as f:
                dic = pickle.load(f)

            head = dic.pop("header", {})
            station_name, meta_row = _extract_station_meta(head)

            # Nome canônico (mesmo usado para gravar o parquet)
            station_key = (station_name or station_from_name).lower()

            # Sanidade: filename vs header
            if station_name and station_name[:4].lower() != station_from_name:
                log.warning(
                    "Mismatch filename/header em %s: '%s' vs '%s'",
                    pkl_path.name, station_from_name, station_name,
                )



            # 3) Skip por falta de coordenadas
            if _discard_stations_without_positions(meta_row):
                skipped_no_pos += 1
                continue

            stations_meta.append(meta_row)

            # 4) Skip por pickle vazio
            if not dic:
                skipped_empty += 1
                log.info("[%d/%d] vazio %s", i, total, pkl_path.name)
                continue

            # Trabalho pesado só agora
            _compute_tec_pcarrier(dic)
            by_const = _group_by_constellation(dic)
            if not by_const:
                skipped_empty += 1
                log.info("[%d/%d] sem dado útil %s", i, total, pkl_path.name)
                continue

            n = _write_station_parquet_merged(
                by_const, tec_pc_folder, station_key, year, doy,
            )
            written_files += n
            if n > 0:
                written_stations += 1
                log.info(
                    "[%d/%d] %d/%03d TEC OK  %s → %d parquet(s)",
                    i, total, year, doy, pkl_path.name, n,
                )

        except Exception:
            shutil.move(str(pkl_path), tec_error_folder / pkl_path.name)
            log.exception(
                "[%d/%d] TEC FAIL %s (movido para tec_error)",
                i, total, pkl_path.name,
            )

    # ---- Resumo ----------------------------------------------------------
    if skipped_already:
        log.info("  → %d de %d estações já estavam processadas (puladas)",
                 skipped_already, total)
    if skipped_no_pos:
        log.info("  → %d estações puladas por falta de coordenadas", skipped_no_pos)
    if skipped_empty:
        log.info("  → %d estações puladas por pickle vazio / sem dado útil", skipped_empty)

    if written_stations:
        log.info("  → %d estações gravadas (%d parquets no total)",
                 written_stations, written_files)
    else:
        log.info("Nada novo para gravar (todas as estações já processadas).")

    # ---- SP file (metadados das estações) -------------------------------
    if stations_meta:
        new_sp_df = pd.DataFrame(
            stations_meta,
            columns=["name", "x_km", "y_km", "z_km", "ln_deg", "lt_deg", "h_km"],
        )
        df_sp = _consolidate_sp(new_sp_df, station_positions_file)
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
        lg.warning("Pasta de pickles inexistente: %s (rode make_rinex_pickes antes)", tec_folder)
        return

    lg.info("=== 3.2 TEC (pcarrier) [layout long particionado] ===")
    process_tec(tec_folder, tec_pc_folder, tec_error_folder, sp_file, year, doy)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    args = build_arg_parser(
        "Etapa 3.2 — Cálculo de TEC (pcarrier) por estação [layout long]"
    ).parse_args()

    pairs = resolve_pairs(args, 'rinexPKLReferenceFolder')
    log.info("Encontradas %d pastas year/doy para processar", len(pairs))
    # pairs = [(2021,295)]
    for year, doy in pairs:
        log.info("=== %d / doy %03d ===", year, doy)
        run(year, doy)