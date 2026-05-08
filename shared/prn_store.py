# -*- coding: utf-8 -*-
"""I/O dos parquets-por-PRN do pipeline de TEC.

Cada arquivo Parquet corresponde a um PRN de um dia ``(year, doy)`` e
contém uma coluna ``horario`` + uma coluna por estação. Após a etapa que
calcula as posições dos satélites, ganha também ``x_km``, ``y_km``, ``z_km``.

Este módulo concentra toda a lógica de leitura/escrita desse formato:
inserção incremental de novas estações, deduplicação por horário e
descoberta rápida de quais estações já estão presentes (lendo apenas o
schema dos parquets, sem carregar os dados). Não é executável.
"""

from __future__ import annotations

import logging
import re
import datetime as dt
from pathlib import Path
import duckdb

import pandas as pd
import pyarrow.parquet as pq
from shared.config import tec_pc_dir, dtec_pc_dir

log = logging.getLogger("process.prn_store")

PRN_RE = re.compile(r"([GRECIJS]\d{2})$")

# Colunas dos parquets por PRN que NÃO são nomes de estação
NON_STATION_COLS = {"horario", "x_km", "y_km", "z_km"}


# --------------------------------------------------------------------------
# Helpers genéricos
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


# --------------------------------------------------------------------------
# Inspeção de schema (rápida — não carrega dados)
# --------------------------------------------------------------------------

def get_processed_stations(tec_pc_folder: Path) -> set[str]:
    """Retorna o conjunto de nomes de estação que já aparecem como coluna
    em pelo menos um parquet por PRN em ``tec_pc_folder``.

    Lê apenas o *schema* dos arquivos Parquet (metadados), não os dados —
    operação rápida mesmo com muitos parquets.
    """
    if not tec_pc_folder.exists():
        return set()

    processed: set[str] = set()
    for pq_path in tec_pc_folder.glob("*.parquet"):
        try:
            schema = pq.read_schema(pq_path)
            processed.update(set(schema.names) - NON_STATION_COLS)
        except Exception as exc:
            log.warning("Erro lendo schema de %s: %s", pq_path.name, exc)

    return processed


# --------------------------------------------------------------------------
# Escrita incremental
# --------------------------------------------------------------------------

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

def load_tec_with_neighbors(
    year: int,
    doy: int,
    station: str,
    constellation: str = "G",
    with_neighbors: bool = True,
) -> pd.DataFrame:
    """Read the central day's TEC, optionally padded with DOY±1 neighbors.

    Returns rows for the central day only — neighbors (when loaded) just
    feed downstream interpolators slack at day boundaries.
    """
    base = dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)

    # -1, 0, +1 with neighbors; only 0 without.
    offsets = (-1, 0, 1) if with_neighbors else (0,)
    pairs = [
        (
            (base + dt.timedelta(days=k)).year,
            (base + dt.timedelta(days=k)).timetuple().tm_yday,
        )
        for k in offsets
    ]

    files = []
    for y, d in pairs:
        path = tec_pc_dir(y, d) / constellation
        files.extend(path.glob(f"{y}{d:03d}{constellation}_{station}.parquet"))

    if not files:
        return pd.DataFrame(columns=["prn", "horario", "tec_pcarrier"])

    # Always filter to the central day; neighbors are read just for padding.
    day_start = dt.datetime(year, 1, 1) + dt.timedelta(days=doy - 1)
    day_end = day_start + dt.timedelta(days=1)

    with duckdb.connect(":memory:") as con:
        df = con.execute(f"""
            SELECT prn, horario, tec_pcarrier
            FROM read_parquet({[str(f) for f in files]!r})
            WHERE horario >= TIMESTAMP '{day_start.isoformat()}'
              AND horario <  TIMESTAMP '{day_end.isoformat()}'
            ORDER BY prn, horario
            """).df()
    return df

# --------------------------------------------------------------------------
# Carregar informações
# --------------------------------------------------------------------------

def get_station_names(year: int, doy: int, extension: str = "parquet") -> list[str]:
    """
    Recursivamente busca arquivos com a extensão especificada e retorna
    uma lista ordenada com o nome das estações (únicas).
    
    O nome da estação é definido como os 4 primeiros caracteres do arquivo.
    """
    # folder = Path(folder)
    folder = tec_pc_dir(year, doy)
    ext = extension if extension.startswith('.') else f'.{extension}'
    ext = ext.lower()

    # Usamos set comprehension para extrair os 4 primeiros caracteres do nome 
    # do arquivo (p.name[:4]) e garantir que não haverá duplicatas.
    stations = {
        p.name[:4] 
        for p in folder.rglob('*') 
        if p.is_file() and p.suffix.lower() == ext
    }
    
    # Retorna a lista de estações ordenada alfabeticamente
    return sorted(list(stations))



def load_tec_with_neighbors_v2(
    year: int,
    doy: int,
    station: str,
    with_neighbors: bool = True,
) -> pd.DataFrame:
    """Read the central day's TEC, optionally padded with DOY±1 neighbors.

    Returns rows for the central day only — neighbors (when loaded) just
    feed downstream interpolators slack at day boundaries.
    """
    base = dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)

    # -1, 0, +1 with neighbors; only 0 without.
    offsets = (-1, 0, 1) if with_neighbors else (0,)
    pairs = [
        (
            (base + dt.timedelta(days=k)).year,
            (base + dt.timedelta(days=k)).timetuple().tm_yday,
        )
        for k in offsets
    ]

    files = []
    for y, d in pairs:
        path = tec_pc_dir(y, d) 
        files.extend(path.rglob(f"{station}_{y}{d:03d}.parquet"))

    if not files:
        return pd.DataFrame(columns=["prn", "horario", "tec_pcarrier"])

    # Always filter to the central day; neighbors are read just for padding.
    day_start = base - dt.timedelta(days=1)
    day_end = base + dt.timedelta(days=1)

    with duckdb.connect(":memory:") as con:
        df = con.execute(f"""
            SELECT prn, horario, tec_pcarrier
            FROM read_parquet({[str(f) for f in files]!r})
            WHERE horario >= TIMESTAMP '{day_start.isoformat()}'
              AND horario <  TIMESTAMP '{day_end.isoformat()}'
            ORDER BY prn, horario
        """).df()
    return df


def load_dtec_with_neighbors(
    year: int,
    doy: int,
    station: str,
    with_neighbors: bool = True,
) -> pd.DataFrame:
    """Read the central day's TEC, optionally padded with DOY±1 neighbors.

    Returns rows for the central day only — neighbors (when loaded) just
    feed downstream interpolators slack at day boundaries.
    """
    base = dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)

    # -1, 0, +1 with neighbors; only 0 without.
    offsets = (-1, 0, 1) if with_neighbors else (0,)
    pairs = [
        (
            (base + dt.timedelta(days=k)).year,
            (base + dt.timedelta(days=k)).timetuple().tm_yday,
        )
        for k in offsets
    ]

    files = []
    for y, d in pairs:
        path = dtec_pc_dir(y, d) 
        files.extend(path.rglob(f"{station}_{y}{d:03d}.parquet"))

    if not files:
        return pd.DataFrame(columns=["prn", "horario", "tec_pcarrier"])

    # Always filter to the central day; neighbors are read just for padding.
    day_start = base - dt.timedelta(days=1)
    day_end = base + dt.timedelta(days=1)

    with duckdb.connect(":memory:") as con:
        df = con.execute(f"""
            SELECT prn, horario, tec_pcarrier
            FROM read_parquet({[str(f) for f in files]!r})
            WHERE horario >= TIMESTAMP '{day_start.isoformat()}'
              AND horario <  TIMESTAMP '{day_end.isoformat()}'
            ORDER BY prn, horario
        """).df()
    return df